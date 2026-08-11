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
}
