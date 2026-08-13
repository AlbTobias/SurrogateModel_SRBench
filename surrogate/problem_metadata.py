"""Auditable feature domains for the deterministic surrogate problems."""

from __future__ import annotations


PROBLEM_DOMAINS: dict[str, dict[str, tuple[float, float]]] = {
    "cantilever": {
        "force": (0.5, 1.5),
        "length": (0.8, 1.2),
        "modulus": (0.8, 1.2),
        "width": (0.7, 1.3),
        "height": (0.7, 1.3),
    },
    "borehole": {
        "borehole_radius": (0.05, 0.15),
        "influence_radius": (100.0, 50000.0),
        "upper_transmissivity": (63070.0, 115600.0),
        "upper_head": (990.0, 1110.0),
        "lower_transmissivity": (63.1, 116.0),
        "lower_head": (700.0, 820.0),
        "borehole_length": (1120.0, 1680.0),
        "hydraulic_conductivity": (9855.0, 12045.0),
    },
    "piston": {
        "piston_mass": (30.0, 60.0),
        "surface_area": (0.005, 0.020),
        "initial_volume": (0.002, 0.010),
        "spring_coefficient": (1000.0, 5000.0),
        "atmospheric_pressure": (90000.0, 110000.0),
        "ambient_temperature": (290.0, 296.0),
        "gas_temperature": (340.0, 360.0),
    },
    "ccpp": {
        "ambient_temperature": (1.81, 37.11),
        "exhaust_vacuum": (25.36, 81.56),
        "ambient_pressure": (992.89, 1033.30),
        "relative_humidity": (25.56, 100.16),
    },
    "naval_propulsion": {
        "ship_speed": (3.0, 27.0),
        "compressor_decay": (0.95, 1.0),
        "turbine_decay": (0.975, 1.0),
    },
    "wing_weight": {
        "wing_area": (150.0, 200.0),
        "fuel_weight": (220.0, 300.0),
        "aspect_ratio": (6.0, 10.0),
        "sweep_angle_degrees": (-10.0, 10.0),
        "dynamic_pressure": (16.0, 45.0),
        "taper_ratio": (0.5, 1.0),
        "thickness_chord_ratio": (0.08, 0.18),
        "ultimate_load_factor": (2.5, 6.0),
        "design_gross_weight": (1700.0, 2500.0),
        "paint_weight": (0.025, 0.08),
    },
}
