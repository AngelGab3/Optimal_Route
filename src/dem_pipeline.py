"""
Pipeline de procesamiento de Modelos de Elevación Digital (DEM) para Marte.
Decodificación Float32, muestreo bilineal O(1) y operador de gradiente discreto.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np


class DEMPipeline:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.metadata = self._load_metadata()
        self.dem = self._load_dem()
        
        self.rows, self.cols = self.metadata["shape"]
        self.width_m, self.depth_m = self.metadata["size_m"]
        self.resolution_m = self.metadata.get("resolution_m", 20.0)
        self.min_elev_m, self.max_elev_m = self.metadata["elevation_range_m"]
        
        self.half_w = self.width_m * 0.5
        self.half_d = self.depth_m * 0.5

    def _load_metadata(self) -> Dict[str, Any]:
        meta_path = self.dataset_dir / "metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_dem(self) -> np.ndarray:
        f32_filename = self.metadata.get("heightmap", {}).get("file", "heightmap.f32")
        f32_path = self.dataset_dir / f32_filename
        rows, cols = self.metadata["shape"]
        
        # Lectura de binario Float32 little-endian
        dem_data = np.fromfile(f32_path, dtype="<f4")
        if dem_data.size != rows * cols:
            raise ValueError(f"Dimensión inesperada en {f32_path}: esperado {rows*cols}, obtenido {dem_data.size}")
        return dem_data.reshape((rows, cols))

    def world_to_grid(self, x_m: float, z_m: float) -> Tuple[float, float]:
        """Convierte coordenadas métricas mundiales (origen en el centro) a coordenadas continuas de celda (c, r)."""
        u = np.clip((x_m + self.half_w) / self.width_m, 0.0, 1.0)
        v = np.clip((z_m + self.half_d) / self.depth_m, 0.0, 1.0)
        c = u * (self.cols - 1)
        r = v * (self.rows - 1)
        return float(c), float(r)

    def grid_to_world(self, c: float, r: float) -> Tuple[float, float]:
        """Convierte índices de rejilla continua (c, r) a coordenadas métricas mundiales (x_m, z_m)."""
        x_m = (c / (self.cols - 1) - 0.5) * self.width_m
        z_m = (r / (self.rows - 1) - 0.5) * self.depth_m
        return float(x_m), float(z_m)

    def sample_elevation_m(self, x_m: float, z_m: float) -> float:
        """Interpolación bilineal O(1) de elevación absoluta en metros."""
        c, r = self.world_to_grid(x_m, z_m)
        c0 = int(np.floor(c))
        c1 = min(c0 + 1, self.cols - 1)
        r0 = int(np.floor(r))
        r1 = min(r0 + 1, self.rows - 1)
        
        fc = c - c0
        fr = r - r0
        
        h00 = self.dem[r0, c0]
        h10 = self.dem[r0, c1]
        h01 = self.dem[r1, c0]
        h11 = self.dem[r1, c1]
        
        h0 = h00 + (h10 - h00) * fc
        h1 = h01 + (h11 - h01) * fc
        return float(h0 + (h1 - h0) * fr)

    def sample_relative_elevation_m(self, x_m: float, z_m: float) -> float:
        """Elevación relativa a la cota mínima del dataset en metros."""
        return self.sample_elevation_m(x_m, z_m) - self.min_elev_m

    def sample_slope_deg(self, x_m: float, z_m: float, delta_step_m: Optional[float] = None) -> float:
        """
        Calcula la pendiente real no exagerada en grados mediante diferencias finitas centrales:
        grad_x = (z(x+dx) - z(x-dx)) / (2*dx)
        grad_z = (z(z+dz) - z(z-dz)) / (2*dz)
        slope = arctan(sqrt(grad_x^2 + grad_z^2))
        """
        step = delta_step_m if delta_step_m is not None else max(1.0, self.resolution_m)
        z_e = self.sample_elevation_m(x_m + step, z_m)
        z_w = self.sample_elevation_m(x_m - step, z_m)
        z_s = self.sample_elevation_m(x_m, z_m + step)
        z_n = self.sample_elevation_m(x_m, z_m - step)
        
        dz_dx = (z_e - z_w) / (2.0 * step)
        dz_dz = (z_s - z_n) / (2.0 * step)
        
        grad_norm = np.hypot(dz_dx, dz_dz)
        slope_rad = np.arctan(grad_norm)
        return float(np.degrees(slope_rad))

    def compute_slope_map(self, target_rows: int = 400, target_cols: int = 400) -> np.ndarray:
        """Genera una matriz 2D de pendientes (en grados) remuestreada para análisis topográfico."""
        xs = np.linspace(-self.half_w, self.half_w, target_cols)
        zs = np.linspace(-self.half_d, self.half_d, target_rows)
        slopes = np.zeros((target_rows, target_cols), dtype=np.float32)
        
        for i, z in enumerate(zs):
            for j, x in enumerate(xs):
                slopes[i, j] = self.sample_slope_deg(x, z)
        return slopes
