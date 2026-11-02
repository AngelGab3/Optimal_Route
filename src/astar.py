"""
Planificador de ruta clásico A* sobre superficies 2.5D de Marte.
Utiliza cola de prioridad min-heap, costo metabólico fisiológico y heurística admisible de O2.
"""

import heapq
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from .dem_pipeline import DEMPipeline
from .physiology import PhysiologicalModel


class AStarPlanner:
    def __init__(self, dem_pipeline: DEMPipeline, physio: PhysiologicalModel, grid_n: int = 160):
        self.dem = dem_pipeline
        self.physio = physio
        self.grid_n = grid_n
        
        self.cell_w_m = self.dem.width_m / (self.grid_n - 1)
        self.cell_d_m = self.dem.depth_m / (self.grid_n - 1)
        
        # Precomputar rejilla discreta de elevación y pendiente
        self.xs = np.linspace(-self.dem.half_w, self.dem.half_w, self.grid_n)
        self.zs = np.linspace(-self.dem.half_d, self.dem.half_d, self.grid_n)
        
        self.node_elev = np.zeros((self.grid_n, self.grid_n), dtype=np.float32)
        self.node_slope = np.zeros((self.grid_n, self.grid_n), dtype=np.float32)
        
        for iz, z in enumerate(self.zs):
            for ix, x in enumerate(self.xs):
                self.node_elev[iz, ix] = self.dem.sample_elevation_m(x, z)
                self.node_slope[iz, ix] = self.dem.sample_slope_deg(x, z)

    def to_grid(self, x_m: float, z_m: float) -> Tuple[int, int]:
        gx = int(np.clip(np.round(((x_m + self.dem.half_w) / self.dem.width_m) * (self.grid_n - 1)), 0, self.grid_n - 1))
        gz = int(np.clip(np.round(((z_m + self.dem.half_d) / self.dem.depth_m) * (self.grid_n - 1)), 0, self.grid_n - 1))
        return gx, gz

    def to_world(self, gx: int, gz: int) -> Tuple[float, float]:
        return float(self.xs[gx]), float(self.zs[gz])

    def plan_route(
        self,
        start_m: Tuple[float, float],
        goal_m: Tuple[float, float],
        thermal_penalty: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """
        Calcula la ruta óptima en A* entre start_m y goal_m minimizando el consumo de O2 y el riesgo.
        """
        sx, sz = self.to_grid(start_m[0], start_m[1])
        gx, gz = self.to_grid(goal_m[0], goal_m[1])
        
        total_nodes = self.grid_n * self.grid_n
        start_idx = sz * self.grid_n + sx
        goal_idx = gz * self.grid_n + gx
        
        if start_idx == goal_idx:
            return {
                "algorithm": "astar",
                "path_m": [list(start_m), list(goal_m)],
                "metrics": {"distance_m": 0.0, "oxygen_l": 0.0, "max_slope_deg": 0.0, "risk_score": 0.0}
            }

        g_score = np.full(total_nodes, np.inf, dtype=np.float64)
        f_score = np.full(total_nodes, np.inf, dtype=np.float64)
        came_from = np.full(total_nodes, -1, dtype=np.int32)
        closed = np.zeros(total_nodes, dtype=bool)

        def heuristic(ix: int, iz: int) -> float:
            # Heurística admisible: mínima distancia euclidiana a la velocidad de marcha y tasa base
            dist = np.hypot((gx - ix) * self.cell_w_m, (gz - iz) * self.cell_d_m)
            return (dist / self.physio.walk_speed_ms) * self.physio.base_rate_l_s * 0.85

        g_score[start_idx] = 0.0
        f_score[start_idx] = heuristic(sx, sz)
        
        # Priority queue: (f_score, node_idx)
        open_heap = [(f_score[start_idx], start_idx)]
        
        # 8 direcciones: dx, dz, dist_m
        dirs = [
            (-1, 0, self.cell_w_m), (1, 0, self.cell_w_m),
            (0, -1, self.cell_d_m), (0, 1, self.cell_d_m),
            (-1, -1, np.hypot(self.cell_w_m, self.cell_d_m)),
            (1, -1, np.hypot(self.cell_w_m, self.cell_d_m)),
            (-1, 1, np.hypot(self.cell_w_m, self.cell_d_m)),
            (1, 1, np.hypot(self.cell_w_m, self.cell_d_m)),
        ]

        while open_heap:
            current_f, current = heapq.heappop(open_heap)
            if current == goal_idx:
                break
            if closed[current]:
                continue
            closed[current] = True
            
            cx = current % self.grid_n
            cz = current // self.grid_n
            cur_elev = self.node_elev[cz, cx]
            
            for dx, dz, step_dist in dirs:
                nx = cx + dx
                nz = cz + dz
                if not (0 <= nx < self.grid_n and 0 <= nz < self.grid_n):
                    continue
                neighbor = nz * self.grid_n + nx
                if closed[neighbor]:
                    continue
                
                next_elev = self.node_elev[nz, nx]
                d_elev = next_elev - cur_elev
                seg_slope = np.degrees(np.arctan(abs(d_elev) / step_dist))
                local_slope = max(float(self.node_slope[nz, nx]), float(seg_slope))
                
                # Costo fisiológico de O2
                o2_cost, _, _ = self.physio.calculate_segment_cost(
                    step_dist, d_elev, local_slope, running=False, thermal_penalty=thermal_penalty
                )
                
                # Penalización severa por exceder la pendiente segura
                slope_penalty = 1.0
                if local_slope > self.physio.max_safe_slope_deg:
                    excess = local_slope - self.physio.max_safe_slope_deg
                    slope_penalty += 30.0 + (excess ** 2) * 8.0
                elif local_slope > self.physio.max_safe_slope_deg * 0.75:
                    r = (local_slope - self.physio.max_safe_slope_deg * 0.75) / (self.physio.max_safe_slope_deg * 0.25)
                    slope_penalty += (r ** 2) * 3.5
                    
                edge_cost = o2_cost * slope_penalty
                tentative_g = g_score[current] + edge_cost
                
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_val = tentative_g + heuristic(nx, nz)
                    f_score[neighbor] = f_val
                    heapq.heappush(open_heap, (f_val, neighbor))

        if g_score[goal_idx] == np.inf:
            return None

        # Reconstruir camino
        curr = goal_idx
        path_indices = [curr]
        while curr != start_idx:
            curr = came_from[curr]
            if curr == -1:
                break
            path_indices.append(curr)
        path_indices.reverse()

        path_m = []
        for idx in path_indices:
            ix = idx % self.grid_n
            iz = idx // self.grid_n
            path_m.append(list(self.to_world(ix, iz)))

        # Asegurar exactitud en extremos
        path_m[0] = list(start_m)
        path_m[-1] = list(goal_m)

        metrics = self.evaluate_path_metrics(path_m, thermal_penalty)
        return {
            "algorithm": "astar",
            "path_m": path_m,
            "metrics": metrics
        }

    def evaluate_path_metrics(self, path_m: List[List[float]], thermal_penalty: float = 1.0) -> Dict[str, float]:
        """Calcula métricas detalladas a lo largo de cualquier polilínea métrica."""
        tot_dist = 0.0
        tot_o2 = 0.0
        max_slope = 0.0
        slope_list = []
        
        for i in range(len(path_m) - 1):
            x0, z0 = path_m[i]
            x1, z1 = path_m[i+1]
            seg_dist = np.hypot(x1 - x0, z1 - z0)
            if seg_dist < 1e-3:
                continue
            tot_dist += seg_dist
            
            e0 = self.dem.sample_elevation_m(x0, z0)
            e1 = self.dem.sample_elevation_m(x1, z1)
            d_elev = e1 - e0
            seg_slope = max(
                self.dem.sample_slope_deg((x0+x1)*0.5, (z0+z1)*0.5),
                np.degrees(np.arctan(abs(d_elev) / seg_dist))
            )
            max_slope = max(max_slope, seg_slope)
            slope_list.append(seg_slope)
            
            o2_seg, _, _ = self.physio.calculate_segment_cost(
                seg_dist, d_elev, seg_slope, running=False, thermal_penalty=thermal_penalty
            )
            tot_o2 += o2_seg

        risk_score = float(np.clip(max_slope / self.physio.max_safe_slope_deg, 0.0, 1.0))
        slope_variance = float(np.var(slope_list)) if slope_list else 0.0
        
        return {
            "distance_m": float(tot_dist),
            "oxygen_l": float(tot_o2),
            "max_slope_deg": float(max_slope),
            "slope_variance": slope_variance,
            "risk_score": risk_score
        }
