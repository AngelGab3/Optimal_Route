"""
Física Solar Marciana y Transferencia Radiativa:
- Distancia heliocéntrica Kepleriana r(Ls) y ley de radiación 1/r².
- Declinación solar y coordenadas horizontales (elevación y acimut).
- Atenuación de Beer-Lambert por columna de polvo MEDA (τ).
"""

from typing import Dict, Any
import numpy as np


class MartianSolarPhysics:
    def __init__(self, lat_deg: float = 18.38, lon_deg: float = 77.58):
        self.lat_deg = lat_deg
        self.lon_deg = lon_deg
        self.a_au = 1.52368
        self.ecc = 0.0934
        self.obliq_deg = 25.19
        self.i_earth_wm2 = 1361.0

    def compute_solar_state(self, ls_deg: float, ltst_h: float, dust_tau: float = 0.45) -> Dict[str, Any]:
        """
        Calcula las variables físicas del Sol para una estación Ls y una hora solar local LTST.
        """
        # 1. Distancia heliocéntrica r(Ls) en AU
        nu_rad = np.radians(ls_deg - 251.0)
        r_au = (self.a_au * (1.0 - self.ecc ** 2)) / (1.0 + self.ecc * np.cos(nu_rad))
        
        # 2. Irradiancia al tope de la atmósfera (TOA)
        i_toa = self.i_earth_wm2 / (r_au ** 2)
        
        # 3. Declinación solar
        sin_delta = np.sin(np.radians(self.obliq_deg)) * np.sin(np.radians(ls_deg))
        delta_rad = np.arcsin(np.clip(sin_delta, -1.0, 1.0))
        delta_deg = np.degrees(delta_rad)
        
        # 4. Ángulo horario y elevación solar
        phi_rad = np.radians(self.lat_deg)
        h_rad = np.radians((ltst_h - 12.0) * 15.0)
        
        sin_alpha = np.sin(phi_rad) * np.sin(delta_rad) + np.cos(phi_rad) * np.cos(delta_rad) * np.cos(h_rad)
        sin_alpha = np.clip(sin_alpha, -1.0, 1.0)
        alpha_rad = np.arcsin(sin_alpha)
        alpha_deg = np.degrees(alpha_rad)
        
        # Acimut
        y_az = -np.cos(delta_rad) * np.sin(h_rad)
        x_az = np.sin(delta_rad) * np.cos(phi_rad) - np.cos(delta_rad) * np.sin(phi_rad) * np.cos(h_rad)
        az_deg = (np.degrees(np.arctan2(y_az, x_az)) + 360.0) % 360.0
        
        # 5. Atenuación atmosférica Beer-Lambert
        if alpha_deg > 0.5:
            airmass = 1.0 / (np.sin(alpha_rad) + 0.095 * (alpha_deg + 3.2) ** (-1.22))
            airmass = float(np.clip(airmass, 1.0, 14.0))
            direct_flux = i_toa * np.sin(alpha_rad) * np.exp(-dust_tau * airmass * 0.68)
            diffuse_flux = i_toa * np.sin(alpha_rad) * 0.34 * (1.0 - np.exp(-dust_tau * airmass * 0.68)) * np.exp(-0.18 * dust_tau)
            surface_irradiance = max(0.0, direct_flux + diffuse_flux)
            phase = "DIA"
        elif alpha_deg > -6.5:
            airmass = 14.0
            tw_factor = (alpha_deg + 6.5) / 7.0
            surface_irradiance = 12.0 * (tw_factor ** 2)
            phase = "CREPUSCULO_AZUL"
        else:
            airmass = 20.0
            surface_irradiance = 0.0
            phase = "NOCHE"
            
        # Estimación de temperatura superficial diurna
        diurnal_wave = np.cos(((ltst_h - 14.2) / 24.0) * np.pi * 2.0)
        mean_temp_c = -56.0 + (1.524 / r_au - 1.0) * 90.0
        temp_c = mean_temp_c + 31.0 * diurnal_wave
        
        return {
            "ls_deg": ls_deg,
            "ltst_h": ltst_h,
            "distance_au": float(r_au),
            "toa_irradiance_wm2": float(i_toa),
            "declination_deg": float(delta_deg),
            "elevation_deg": float(alpha_deg),
            "azimuth_deg": float(az_deg),
            "surface_irradiance_wm2": float(surface_irradiance),
            "airmass": airmass,
            "phase": phase,
            "temp_surface_c": float(temp_c),
        }
