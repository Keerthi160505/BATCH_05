"""
Unit Tests for Resource & Energy Profiler
Validates:
- HardwarePowerModel power scaling and TDP bounds
- Asynchronous sampling of CPU, process RAM, and system RAM
- Prevention of negative RAM delta values
- Riemann integration of power over time to compute Joules and kWh
- Carbon emissions calculation using documented grid intensity
"""

import time
import pytest
from src.profiler import ResourceEnergyProfiler, HardwarePowerModel


def test_hardware_power_model():
    model = HardwarePowerModel(tdp_watts=65.0, idle_ratio=0.20)
    p_idle = model.calculate_power(0.0)
    p_full = model.calculate_power(100.0)
    p_mid = model.calculate_power(50.0)

    assert p_idle < p_mid < p_full
    assert p_idle >= 13.0
    assert p_full >= 65.0
    assert model.carbon_intensity_g_per_kwh == 475.0


def test_profiler_timing_and_metrics():
    profiler = ResourceEnergyProfiler(sampling_interval_s=0.01, use_codecarbon=False)
    profiler.start()

    # Perform reproducible computational workload
    total = sum(i * i for i in range(300000))
    time.sleep(0.06)
    stats = profiler.stop()

    assert stats["duration_s"] >= 0.05
    assert stats["energy_joules"] > 0.0
    assert stats["energy_kwh"] > 0.0
    assert stats["co2_emissions_g"] >= 0.0
    assert stats["peak_ram_mb"] >= stats["baseline_ram_mb"]
    assert stats["delta_ram_mb"] >= 0.0, "Delta RAM must never be negative"
    assert stats["is_energy_estimated"] is True
    assert "TDP-calibrated" in stats["power_measurement_method"]
    assert stats["gpu_supported"] is False
