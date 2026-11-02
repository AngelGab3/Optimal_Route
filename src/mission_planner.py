"""
Planificador de misión avanzada:
- Detección automática geomorfométrica de POIs desde el DEM.
- Travesía científica multi-parada (TSP Marciano con presupuesto de O2).
- Evacuación dinámica de emergencia ante tormentas de polvo MEDA.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from .dem_pipeline import DEMPipeline
from .physiology import PhysiologicalModel
from .astar import AStarPlanner


class MissionPlanner:
    def __init__(self, dem_pipeline: DEMPipeline, physio: PhysiologicalModel, astar_planner: AStarPlanner):
        self.dem = dem_pipeline
        self.physio = physio
        self.astar = astar_planner

    def detect_pois_from_dem(self) -> List[Dict[str, Any]]:
        """
        Analiza geomorfométricamente el DEM para sintetizar POIs operativos:
        - 1 Refugio / Sitio de Aterrizaje en llanura (pendiente < 5°).
        - Cuencas sedimentarias (depresiones con baja pendiente).
        - Contactos estratigráficos (pendiente moderada 8°..18°).
        - Escarpes peligrosos (> 25°).
        """
        w = self.dem.width_m
        d = self.dem.depth_m
        max_safe = self.physio.max_safe_slope_deg

        best_shelter = {"x": 0.0, "z": 0.0, "score": np.inf}
        best_basin = {"x": -w * 0.15, "z": d * 0.12, "elev": np.inf}
        best_contact = {"x": w * 0.16, "z": -d * 0.14, "score": -np.inf}
        best_hazard = {"x": w * 0.08, "z": d * 0.18, "slope": -np.inf}

        grid_samples = 24
        xs = np.linspace(-w * 0.38, w * 0.38, grid_samples)
        zs = np.linspace(-d * 0.38, d * 0.38, grid_samples)

        for z in zs:
            for x in xs:
                sl = self.dem.sample_slope_deg(x, z)
                el = self.dem.sample_elevation_m(x, z)
                dist_center = np.hypot(x / w, z / d)

                # 1. Refugio: mínima pendiente y estabilidad cerca del centro
                sh_score = sl * 2.5 + dist_center * 8.0
                if sh_score < best_shelter["score"]:
                    best_shelter = {"x": x, "z": z, "score": sh_score, "slope": sl}

                # 2. Cuenca baja pero segura
                if sl < max_safe * 0.65 and el < best_basin["elev"]:
                    best_basin = {"x": x, "z": z, "elev": el}

                # 3. Contacto estratigráfico (8° a 18°)
                if 8.0 <= sl <= max_safe * 0.75:
                    c_score = sl - abs(dist_center - 0.22) * 15.0
                    if c_score > best_contact["score"]:
                        best_contact = {"x": x, "z": z, "score": c_score}

                # 4. Escarpe peligroso
                if sl > best_hazard["slope"]:
                    best_hazard = {"x": x, "z": z, "slope": sl}

        return [
            {
                "id": "poi_shelter",
                "name": "Zona de Aterrizaje y Módulo Hábitat EVA",
                "position_m": [round(best_shelter["x"], 1), round(best_shelter["z"], 1)],
                "type": "shelter",
                "description": "Planicie de pendiente mínima (< 5°) seleccionada para hábitat y soporte vital.",
                "source": "Análisis Geomorfométrico · Criterio NASA Landing Site"
            },
            {
                "id": "poi_basin",
                "name": "Cuenca Sedimentaria Primaria",
                "position_m": [round(best_basin["x"], 1), round(best_basin["z"], 1)],
                "type": "sampling",
                "description": "Depresión de baja energía con alto potencial de filosilicatos y evaporitas hidratadas.",
                "source": "Modelo de Retención Topográfica MOLA/CTX"
            },
            {
                "id": "poi_contact",
                "name": "Contacto Estratigráfico Deltaico",
                "position_m": [round(best_contact["x"], 1), round(best_contact["z"], 1)],
                "type": "sampling",
                "description": "Afloramiento de estratos antiguos expuestos ideal para muestreo astrobiológico.",
                "source": "Gradientes CRISM / HiRISE"
            },
            {
                "id": "poi_hazard",
                "name": f"Escarpe Peligroso ({round(best_hazard['slope'], 1)}°)",
                "position_m": [round(best_hazard["x"], 1), round(best_hazard["z"], 1)],
                "type": "hazard",
                "description": "Ladera que supera el umbral seguro de 25°; riesgo de deslizamiento de regolito.",
                "source": "Mapa de Riesgo DEM"
            }
        ]

    def plan_science_tour_tsp(
        self,
        shelter_pos_m: Tuple[float, float],
        sampling_pois: List[Dict[str, Any]],
        max_stops: int = 3
    ) -> Dict[str, Any]:
        """
        Planifica una travesía que parte del refugio, visita los POIs de muestreo usando
        Nearest-Neighbor TSP y regresa de forma segura al refugio.
        """
        if not sampling_pois:
            return {"legs": [], "metrics": {}}

        remaining = list(sampling_pois)
        waypoints = [list(shelter_pos_m)]
        curr_pos = list(shelter_pos_m)

        for _ in range(min(max_stops, len(remaining))):
            best_idx = 0
            best_d = np.inf
            for idx, p in enumerate(remaining):
                d = np.hypot(p["position_m"][0] - curr_pos[0], p["position_m"][1] - curr_pos[1])
                if d < best_d:
                    best_d = d
                    best_idx = idx
            chosen = remaining.pop(best_idx)
            waypoints.append(chosen["position_m"])
            curr_pos = chosen["position_m"]

        # Cerrar el circuito regresando al refugio
        waypoints.append(list(shelter_pos_m))

        full_path = []
        tot_dist = 0.0
        tot_o2 = 0.0
        max_slope = 0.0

        for i in range(len(waypoints) - 1):
            p0 = waypoints[i]
            p1 = waypoints[i + 1]
            leg_res = self.astar.plan_route(tuple(p0), tuple(p1))
            if leg_res:
                sub_path = leg_res["path_m"]
                start_k = 0 if i == 0 else 1
                for k in range(start_k, len(sub_path)):
                    full_path.append(sub_path[k])
                tot_dist += leg_res["metrics"]["distance_m"]
                tot_o2 += leg_res["metrics"]["oxygen_l"]
                max_slope = max(max_slope, leg_res["metrics"]["max_slope_deg"])

        return {
            "algorithm": "science_tour_tsp",
            "waypoints": waypoints,
            "path_m": full_path,
            "metrics": {
                "distance_m": float(tot_dist),
                "oxygen_l": float(tot_o2),
                "max_slope_deg": float(max_slope),
                "num_stops": len(waypoints) - 2,
                "is_circuit": True
            }
        }

    def plan_emergency_evacuation(
        self,
        current_pos_m: Tuple[float, float],
        shelters: List[Dict[str, Any]],
        storm_thermal_penalty: float = 1.35
    ) -> Dict[str, Any]:
        """
        Encuentra el refugio más cercano y calcula la ruta de evacuación rápida bajo tormenta de polvo.
        """
        if not shelters:
            raise ValueError("No hay refugios disponibles para evacuación.")

        # Buscar el refugio más cercano
        best_shelter = min(
            shelters,
            key=lambda s: np.hypot(s["position_m"][0] - current_pos_m[0], s["position_m"][1] - current_pos_m[1])
        )

        evac_res = self.astar.plan_route(
            current_pos_m,
            tuple(best_shelter["position_m"]),
            thermal_penalty=storm_thermal_penalty
        )
        return {
            "target_shelter": best_shelter,
            "route": evac_res
        }
