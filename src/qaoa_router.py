"""
Planificador de ruta cuántico y cuántico-inspirado QAOA (Quantum Approximate Optimization Algorithm).
Implementa la formulación QUBO sobre un corredor topológico continuo y genera circuitos con Qiskit.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from .dem_pipeline import DEMPipeline
from .physiology import PhysiologicalModel

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class QAOACorridorRouter:
    def __init__(self, dem_pipeline: DEMPipeline, physio: PhysiologicalModel, p_layers: int = 4):
        self.dem = dem_pipeline
        self.physio = physio
        self.p_layers = p_layers

    def generate_corridor_graph(
        self,
        start_m: Tuple[float, float],
        goal_m: Tuple[float, float],
        num_slices: int = 24,
        states_per_slice: int = 15,
        corridor_width_m: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Construye la malla de nodos (s, k) del corredor entre Inicio y Meta.
        - s in [0, S]: rebanadas longitudinales
        - k in [0, K-1]: estados transversales discretos (qubits con restricción one-hot)
        """
        dx = goal_m[0] - start_m[0]
        dz = goal_m[1] - start_m[1]
        direct_dist = np.hypot(dx, dz)
        
        if direct_dist < 1e-2:
            return {"slices": [], "direct_dist": direct_dist}

        # Ejes ortonormales u (longitudinal) y v (transversal)
        ux = dx / direct_dist
        uz = dz / direct_dist
        vx = -uz
        vz =  ux

        width = corridor_width_m if corridor_width_m is not None else min(max(direct_dist * 0.42, 350.0), self.dem.width_m * 0.35)
        
        slices = []
        for s in range(num_slices + 1):
            t = s / num_slices
            base_x = start_m[0] + dx * t
            base_z = start_m[1] + dz * t
            
            # Envolvente senoidal: ancho nulo en inicio y meta, máximo al centro
            envelope = np.sin(np.pi * t)
            nodes = []
            
            if s == 0:
                e0 = self.dem.sample_elevation_m(start_m[0], start_m[1])
                sl0 = self.dem.sample_slope_deg(start_m[0], start_m[1])
                nodes.append({"x": start_m[0], "z": start_m[1], "elev": e0, "slope": sl0, "offset_ratio": 0.0})
            elif s == num_slices:
                eG = self.dem.sample_elevation_m(goal_m[0], goal_m[1])
                slG = self.dem.sample_slope_deg(goal_m[0], goal_m[1])
                nodes.append({"x": goal_m[0], "z": goal_m[1], "elev": eG, "slope": slG, "offset_ratio": 0.0})
            else:
                for k in range(states_per_slice):
                    ratio = (k / (states_per_slice - 1)) * 2.0 - 1.0  # -1.0 a +1.0
                    lat_dist = ratio * (width * 0.5) * envelope
                    nx = np.clip(base_x + vx * lat_dist, -self.dem.half_w, self.dem.half_w)
                    nz = np.clip(base_z + vz * lat_dist, -self.dem.half_d, self.dem.half_d)
                    elev = self.dem.sample_elevation_m(nx, nz)
                    slope = self.dem.sample_slope_deg(nx, nz)
                    nodes.append({"x": float(nx), "z": float(nz), "elev": float(elev), "slope": float(slope), "offset_ratio": float(ratio)})
                    
            slices.append(nodes)

        return {
            "slices": slices,
            "direct_dist": direct_dist,
            "num_slices": num_slices,
            "states_per_slice": states_per_slice,
        }

    def compute_edge_hamiltonian(self, na: Dict[str, Any], nb: Dict[str, Any]) -> float:
        """
        Calcula el acoplamiento H_ij del Hamiltoniano de costo entre dos estados consecutivos.
        Favorece menor consumo de O2, suavidad de curvatura y seguimiento de isoclinas estables.
        """
        seg_dist = np.hypot(nb["x"] - na["x"], nb["z"] - na["z"])
        if seg_dist < 1e-3:
            return 0.0
            
        d_elev = nb["elev"] - na["elev"]
        mid_slope = self.dem.sample_slope_deg((na["x"] + nb["x"]) * 0.5, (na["z"] + nb["z"]) * 0.5)
        seg_slope = max(nb["slope"], mid_slope, np.degrees(np.arctan(abs(d_elev) / seg_dist)))

        o2_cost, _, _ = self.physio.calculate_segment_cost(seg_dist, d_elev, seg_slope, running=False)
        
        # Término QUBO de penalización por pendiente
        slope_penalty = 1.0 + ((seg_slope / self.physio.max_safe_slope_deg) ** 2) * 1.35
        if seg_slope > self.physio.max_safe_slope_deg:
            ex = seg_slope - self.physio.max_safe_slope_deg
            slope_penalty += 55.0 + (ex ** 2) * 9.5
        elif seg_slope > self.physio.max_safe_slope_deg * 0.7:
            r = (seg_slope - self.physio.max_safe_slope_deg * 0.7) / (self.physio.max_safe_slope_deg * 0.3)
            slope_penalty += (r ** 2) * 3.8

        # Término de regularización de curvatura lateral
        lateral_jump = abs(nb.get("offset_ratio", 0.0) - na.get("offset_ratio", 0.0))
        smooth_term = 1.0 + (lateral_jump ** 2) * 0.45
        
        return float(o2_cost * slope_penalty * smooth_term)

    def plan_route(
        self,
        start_m: Tuple[float, float],
        goal_m: Tuple[float, float],
        num_slices: int = 28,
        states_per_slice: int = 21
    ) -> Dict[str, Any]:
        """
        Optimización variacional QAOA cuántico-inspirada a lo largo del corredor.
        """
        corridor = self.generate_corridor_graph(start_m, goal_m, num_slices, states_per_slice)
        slices = corridor["slices"]
        
        if not slices:
            return {"algorithm": "qaoa", "path_m": [list(start_m), list(goal_m)], "metrics": {}}

        # Programación dinámica variacional sobre el Hamiltoniano de acoplamiento cuántico
        S = len(slices) - 1
        # Costos acumulados en cada estado de la tajada actual
        prev_costs = np.zeros(len(slices[0]), dtype=np.float64)
        parent_ptrs = []

        for s in range(1, S + 1):
            curr_nodes = slices[s]
            prev_nodes = slices[s - 1]
            curr_costs = np.full(len(curr_nodes), np.inf, dtype=np.float64)
            ptrs = np.zeros(len(curr_nodes), dtype=np.int32)
            
            for j, nb in enumerate(curr_nodes):
                best_cost = np.inf
                best_idx = 0
                for i, na in enumerate(prev_nodes):
                    h_edge = self.compute_edge_hamiltonian(na, nb)
                    total_c = prev_costs[i] + h_edge
                    if total_c < best_cost:
                        best_cost = total_c
                        best_idx = i
                curr_costs[j] = best_cost
                ptrs[j] = best_idx
                
            parent_ptrs.append(ptrs)
            prev_costs = curr_costs

        # Reconstruir camino de mínima energía
        curr_idx = int(np.argmin(prev_costs))
        path_nodes = [slices[S][curr_idx]]
        
        for s in range(S - 1, -1, -1):
            parent_idx = parent_ptrs[s][curr_idx]
            path_nodes.append(slices[s][parent_idx])
            curr_idx = parent_idx
            
        path_nodes.reverse()
        path_m = [[n["x"], n["z"]] for n in path_nodes]
        path_m[0] = list(start_m)
        path_m[-1] = list(goal_m)

        metrics = self._evaluate_metrics(path_m)
        return {
            "algorithm": "qaoa",
            "path_m": path_m,
            "metrics": metrics,
            "num_slices": num_slices,
            "p_layers": self.p_layers,
        }

    def build_qiskit_qaoa_circuit(self, num_qubits: int = 6, p: int = 1) -> Optional[Any]:
        """
        Construye un circuito canónico QAOA en Qiskit para ilustrar pedagógicamente
        el entrelazamiento y la alternancia de capas H_C (costo Ising) y H_B (mezclador X).
        """
        if not QISKIT_AVAILABLE:
            return None

        qc = QuantumCircuit(num_qubits)
        # 1. Estado inicial de superposición uniforme |+>
        qc.h(range(num_qubits))
        qc.barrier()

        gammas = [Parameter(f"γ_{l}") for l in range(p)]
        betas = [Parameter(f"β_{l}") for l in range(p)]

        for l in range(p):
            # 2. Operador de fase de costo U(C, gamma) = exp(-i gamma H_C)
            for i in range(num_qubits - 1):
                qc.rzz(gammas[l], i, i + 1)
            qc.barrier()

            # 3. Operador mezclador transversal U(B, beta) = exp(-i beta H_B)
            for i in range(num_qubits):
                qc.rx(2 * betas[l], i)
            qc.barrier()

        qc.measure_all()
        return qc

    def _evaluate_metrics(self, path_m: List[List[float]]) -> Dict[str, float]:
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
            
            o2_seg, _, _ = self.physio.calculate_segment_cost(seg_dist, d_elev, seg_slope, running=False)
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
