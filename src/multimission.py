"""
Fusión sensorial y derivación de capas científicas multi-misión:
- CRISM (MRO): Mineralogía e hidratación (filosilicatos Fe/Mg vs. carbonatos).
- THEMIS (Mars Odyssey): Inercia térmica y regolito (roca consolidada vs. trampas de arena).
- Mapa de Riesgo Geotécnico de Terreno.
"""

from typing import Tuple, Dict, Any
import numpy as np
from .dem_pipeline import DEMPipeline


class MultiMissionLayerProcessor:
    def __init__(self, dem_pipeline: DEMPipeline):
        self.dem_pipeline = dem_pipeline
        self.range_m = max(1.0, self.dem_pipeline.max_elev_m - self.dem_pipeline.min_elev_m)

    def evaluate_point_science(self, x_m: float, z_m: float) -> Dict[str, Any]:
        """Evalúa los índices geofísicos y mineralógicos en un punto métrico (x_m, z_m)."""
        h_rel_m = self.dem_pipeline.sample_relative_elevation_m(x_m, z_m)
        elev_norm = np.clip(h_rel_m / self.range_m, 0.0, 1.0)
        slope_deg = self.dem_pipeline.sample_slope_deg(x_m, z_m)

        # 1. Capa CRISM (MRO)
        # Carbonatos marginales (elevNorm ~ 0.25..0.52 y pendiente moderada 3°..16°)
        carb_score = np.exp(-((elev_norm - 0.36) / 0.14) ** 2) * np.exp(-((slope_deg - 8.5) / 7.5) ** 2)
        # Filosilicatos Fe-Mg en fondo lacustre / abanico deltaico (elevNorm < 0.35, baja pendiente)
        clay_score = np.exp(-((elev_norm - 0.18) / 0.14) ** 2) * np.exp(-((slope_deg) / 9.0) ** 2)

        mineral_type = "Regolito Basáltico Común"
        if carb_score > 0.28 and carb_score >= clay_score:
            mineral_type = "Carbonatos Marginales Hidratados (CRISM)"
        elif clay_score > 0.22:
            mineral_type = "Filosilicatos / Arcillas Fe-Mg (CRISM)"

        # 2. Capa THEMIS (Mars Odyssey) - Inercia Térmica
        ti_norm = np.clip(slope_deg / 24.0 * 0.65 + elev_norm * 0.42, 0.0, 1.0)
        if ti_norm < 0.35:
            regolith_type = "Duna Eólica / Arena Suelta (Baja Inercia Térmica)"
            trafficability = "Riesgo de atrapamiento de ruedas / resbalamiento de bota"
        elif ti_norm > 0.58:
            regolith_type = "Roca Madre Consolidada (Alta Inercia Térmica)"
            trafficability = "Superficie de alta tracción y soporte"
        else:
            regolith_type = "Regolito Mixto Compacto"
            trafficability = "Tránsito nominal"

        # 3. Índice de Riesgo (0.0 a 1.0)
        safe_deg = 25.0
        risk_score = float(np.clip(slope_deg / safe_deg, 0.0, 1.0))
        if slope_deg > safe_deg:
            risk_score += (slope_deg - safe_deg) / 10.0

        return {
            "elevation_m": self.dem_pipeline.sample_elevation_m(x_m, z_m),
            "slope_deg": slope_deg,
            "crism": {
                "clay_score": float(clay_score),
                "carbonate_score": float(carb_score),
                "classification": mineral_type,
            },
            "themis": {
                "thermal_inertia_index": float(ti_norm),
                "classification": regolith_type,
                "trafficability": trafficability,
            },
            "risk_score": float(risk_score),
        }

    def generate_layer_matrices(self, rows: int = 200, cols: int = 200) -> Dict[str, np.ndarray]:
        """Genera matrices 2D de análisis mineralógico, inercia térmica y riesgo sobre todo el dominio."""
        xs = np.linspace(-self.dem_pipeline.half_w, self.dem_pipeline.half_w, cols)
        zs = np.linspace(-self.dem_pipeline.half_d, self.dem_pipeline.half_d, rows)
        
        clay_mat = np.zeros((rows, cols), dtype=np.float32)
        carb_mat = np.zeros((rows, cols), dtype=np.float32)
        ti_mat = np.zeros((rows, cols), dtype=np.float32)
        risk_mat = np.zeros((rows, cols), dtype=np.float32)
        
        for i, z in enumerate(zs):
            for j, x in enumerate(xs):
                pt = self.evaluate_point_science(x, z)
                clay_mat[i, j] = pt["crism"]["clay_score"]
                carb_mat[i, j] = pt["crism"]["carbonate_score"]
                ti_mat[i, j] = pt["themis"]["thermal_inertia_index"]
                risk_mat[i, j] = pt["risk_score"]
                
        return {
            "crism_clays": clay_mat,
            "crism_carbonates": carb_mat,
            "themis_thermal_inertia": ti_mat,
            "risk_index": risk_mat,
        }
