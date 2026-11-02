"""
Modelo bioenergético y fisiológico del astronauta en traje EVA (PLSS).
Basado en adaptaciones de la ecuación de Pandolf/Margaria para gravedad marciana (3.71 m/s²).
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple


class PhysiologicalModel:
    def __init__(self, config_path: Path):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        self.gravity_ms2 = self.config.get("gravity_ms2", 3.71)
        self.eye_height_m = self.config.get("eye_height_m", 1.6)
        self.walk_speed_ms = self.config.get("walk_speed_ms", 1.2)
        self.run_speed_ms = self.config.get("run_speed_ms", 2.5)
        self.max_safe_slope_deg = self.config.get("max_safe_slope_deg", 25.0)
        self.o2_reserve_l = self.config.get("o2_reserve_l", 580.0)
        self.o2_base_consumption_l_per_h = self.config.get("o2_base_consumption_l_per_h", 50.0)
        self.slope_cost_factors = self.config.get("slope_cost_factors", [
            {"from_deg": 0, "to_deg": 5, "factor": 1.0},
            {"from_deg": 5, "to_deg": 10, "factor": 1.35},
            {"from_deg": 10, "to_deg": 15, "factor": 1.85},
            {"from_deg": 15, "to_deg": 20, "factor": 2.6},
            {"from_deg": 20, "to_deg": 25, "factor": 3.8},
            {"from_deg": 25, "to_deg": 90, "factor": 6.5},
        ])
        self.base_rate_l_s = self.o2_base_consumption_l_per_h / 3600.0

    def get_slope_cost_factor(self, slope_deg: float, delta_elev_m: float = 0.0) -> float:
        """Determina el factor multiplicador según la tabla empírica de pendiente."""
        s = max(0.0, slope_deg)
        factor = 1.0
        for entry in self.slope_cost_factors:
            if s >= entry["from_deg"] and s < entry["to_deg"]:
                factor = entry["factor"]
                break
        if s >= self.slope_cost_factors[-1]["from_deg"]:
            factor = self.slope_cost_factors[-1]["factor"]
            
        # Si es descenso pronunciado, hay una reducción de esfuerzo hasta que la inclinación obliga a frenar
        if delta_elev_m < -0.1 and s <= 18.0:
            factor *= 0.82
        elif delta_elev_m < -0.1 and s > 18.0:
            # En descenso escarpado el frenado muscular aumenta el gasto metabólico
            factor *= 1.25
            
        return factor

    def calculate_segment_cost(
        self,
        dist_m: float,
        delta_elev_m: float,
        slope_deg: float,
        running: bool = False,
        thermal_penalty: float = 1.0
    ) -> Tuple[float, float, float]:
        """
        Calcula:
        - o2_consumed_l: Litros de O2 consumidos en el tramo
        - time_s: Tiempo en segundos para recorrer el tramo
        - rate_l_h: Tasa metabólica instantánea equivalente (L/h)
        """
        speed_ms = self.run_speed_ms if running else self.walk_speed_ms
        speed_ms = max(0.1, speed_ms)
        time_s = dist_m / speed_ms
        
        slope_factor = self.get_slope_cost_factor(slope_deg, delta_elev_m)
        climb_grade = delta_elev_m / max(0.1, dist_m)
        
        # Factor de trabajo contra la gravedad marciana
        if climb_grade > 0:
            climb_bonus = 1.0 + climb_grade * 4.5
        else:
            climb_bonus = max(0.75, 1.0 + climb_grade * 0.8)
            
        run_multiplier = 2.2 if running else 1.0
        
        effective_rate_l_s = (
            self.base_rate_l_s * slope_factor * climb_bonus * run_multiplier * thermal_penalty
        )
        o2_consumed_l = effective_rate_l_s * time_s
        rate_l_h = effective_rate_l_s * 3600.0
        
        return o2_consumed_l, time_s, rate_l_h
