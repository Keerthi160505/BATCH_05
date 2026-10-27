"""
Resource & Energy Profiler
Measures and estimates:
- CPU Utilization (% process and % system)
- Memory Footprint (process RSS baseline, peak RAM in MB, non-negative delta RAM in MB, and system memory)
- Execution Duration (training latency, inference latency in seconds)
- Energy Consumption (Joules, kWh) via dimensionally correct Riemann numerical integration of TDP power model
- Carbon Emissions (gCO2eq) based on documented grid carbon intensity
- Explicit hardware metadata (CPU cores, TDP, RAM capacity, GPU scope declaration)
"""

import os
import time
import psutil
import platform
import threading
import warnings
import numpy as np
from typing import Dict, Any, Optional, List

# Suppress CodeCarbon verbose logging during benchmark loops
os.environ["CODECARBON_LOG_LEVEL"] = "ERROR"

try:
    from codecarbon import OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False


def get_cpu_brand_name() -> str:
    """Safely retrieves the processor model name."""
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        return info.get("brand_raw", platform.processor()) or platform.processor() or "Generic x86_64"
    except Exception:
        return platform.processor() or "Generic CPU"


class HardwarePowerModel:
    """
    Empirical Hardware Power Model based on Thermal Design Power (TDP)
    and dynamic processor utilization, standard in Green AI literature:
    P_total(t) = P_idle + (U_cpu(t) / 100) * (P_tdp - P_idle) + P_dram
    E(Joules) = integral P(t) dt

    Assumptions & Calibration:
    - Base idle power is estimated at 20% of TDP (P_idle = 0.20 * P_tdp).
    - Dynamic RAM power consumption is modeled at 0.375 W per GB of allocated RAM.
    - Global average carbon intensity of electricity grid is calibrated to 475 gCO2eq/kWh.
    - This provides an empirical power estimate; it is not a direct physical hardware shunt measurement.
    """
    def __init__(self, tdp_watts: Optional[float] = None, idle_ratio: float = 0.20, dram_watts_per_gb: float = 0.375):
        self.idle_ratio = idle_ratio
        self.dram_watts_per_gb = dram_watts_per_gb

        # Infer TDP if not provided (typical modern x86 desktop/server is 65W - 105W; laptop is 28W - 45W)
        if tdp_watts is None:
            phys_cores = psutil.cpu_count(logical=False) or 4
            self.tdp_watts = float(max(35.0, min(125.0, phys_cores * 12.5)))
        else:
            self.tdp_watts = float(tdp_watts)

        self.p_idle = self.tdp_watts * self.idle_ratio
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        self.p_dram = total_ram_gb * self.dram_watts_per_gb
        self.carbon_intensity_g_per_kwh = 475.0  # Standard global average grid intensity

    def calculate_power(self, cpu_utilization_pct: float) -> float:
        """Instantaneous power draw in Watts."""
        u = max(0.0, min(100.0, float(cpu_utilization_pct))) / 100.0
        return self.p_idle + u * (self.tdp_watts - self.p_idle) + self.p_dram


class ResourceEnergyProfiler:
    """
    Asynchronous resource and energy profiler.
    Samples CPU and memory utilization during code execution.
    """
    def __init__(self, sampling_interval_s: float = 0.02, use_codecarbon: bool = True):
        self.sampling_interval_s = float(sampling_interval_s)
        self.use_codecarbon = bool(use_codecarbon and CODECARBON_AVAILABLE)
        self.power_model = HardwarePowerModel()
        self.process = psutil.Process()

        self._stop_event = threading.Event()
        self._sample_thread: Optional[threading.Thread] = None
        self._samples: List[tuple] = []
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._baseline_rss_mb: float = 0.0
        self._tracker = None

    def _sample_loop(self):
        """High-frequency background sampling loop."""
        while not self._stop_event.is_set():
            try:
                t = time.perf_counter()
                cpu_sys = psutil.cpu_percent(interval=None)
                cpu_proc = self.process.cpu_percent(interval=None) / (psutil.cpu_count() or 1)
                mem_rss = self.process.memory_info().rss / (1024 * 1024)  # MB
                sys_mem_pct = psutil.virtual_memory().percent
                self._samples.append((t, cpu_sys, cpu_proc, mem_rss, sys_mem_pct))
            except Exception:
                pass
            time.sleep(self.sampling_interval_s)

    def start(self):
        """Start tracking hardware resources and energy."""
        self._samples = []
        self._stop_event.clear()

        # Prime psutil CPU counters
        self.process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        self._baseline_rss_mb = self.process.memory_info().rss / (1024 * 1024)

        # CodeCarbon tracker
        if self.use_codecarbon:
            try:
                self._tracker = OfflineEmissionsTracker(
                    country_iso_code="IND",
                    log_level="error",
                    save_to_file=False
                )
                self._tracker.start()
            except Exception:
                self._tracker = None

        self._start_time = time.perf_counter()
        self._sample_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._sample_thread.start()

    def stop(self) -> Dict[str, Any]:
        """Stop tracking and compute comprehensive resource & energy metrics."""
        self._end_time = time.perf_counter()
        self._stop_event.set()
        if self._sample_thread is not None:
            self._sample_thread.join(timeout=1.0)

        duration_s = max(1e-4, self._end_time - self._start_time)

        # CodeCarbon results extraction
        def _extract_float(val: Any) -> float:
            if val is None:
                return 0.0
            if hasattr(val, "kWh"):
                try:
                    return float(val.kWh)
                except Exception:
                    pass
            if hasattr(val, "kg"):
                try:
                    return float(val.kg)
                except Exception:
                    pass
            try:
                return float(val)
            except Exception:
                return 0.0

        cc_emissions_kg = 0.0
        cc_energy_kwh = 0.0
        if self._tracker is not None:
            try:
                raw_emissions = self._tracker.stop()
                cc_emissions_kg = _extract_float(raw_emissions)
                raw_energy = getattr(self._tracker, '_total_energy', None)
                if raw_energy is None:
                    final_data = getattr(self._tracker, 'final_emissions_data', None)
                    if final_data and hasattr(final_data, 'energy_consumed'):
                        raw_energy = final_data.energy_consumed
                cc_energy_kwh = _extract_float(raw_energy)
            except Exception:
                pass

        # Hardware Model Energy & Utilization Calculation via Riemann Sum
        if self._samples:
            cpu_sys_vals = [s[1] for s in self._samples]
            cpu_proc_vals = [s[2] for s in self._samples]
            mem_vals = [s[3] for s in self._samples]

            avg_cpu_pct = float(np.mean(cpu_sys_vals))
            avg_proc_cpu_pct = float(np.mean(cpu_proc_vals))
            peak_cpu_pct = float(max(cpu_sys_vals))
            peak_mem_mb = float(max(mem_vals))
            delta_mem_mb = max(0.0, peak_mem_mb - self._baseline_rss_mb)

            # Numerical Riemann Integration for Energy: E = sum(P_i * dt_i)
            energy_joules = 0.0
            prev_t = self._start_time
            for (t, cpu_sys, _, _, _) in self._samples:
                dt = t - prev_t
                if dt > 0:
                    p = self.power_model.calculate_power(cpu_sys)
                    energy_joules += p * dt
                prev_t = t
            # Final interval
            if self._end_time > prev_t:
                dt = self._end_time - prev_t
                p = self.power_model.calculate_power(cpu_sys_vals[-1])
                energy_joules += p * dt
        else:
            # Short execution fallback
            current_cpu = float(psutil.cpu_percent(interval=None))
            avg_cpu_pct = current_cpu
            avg_proc_cpu_pct = current_cpu
            peak_cpu_pct = current_cpu
            current_rss = float(self.process.memory_info().rss / (1024 * 1024))
            peak_mem_mb = current_rss
            delta_mem_mb = max(0.0, peak_mem_mb - self._baseline_rss_mb)

            p = self.power_model.calculate_power(current_cpu)
            energy_joules = p * duration_s

        # Energy conversions
        energy_kwh = energy_joules / (3600.0 * 1000.0)
        co2_g = energy_kwh * self.power_model.carbon_intensity_g_per_kwh

        # Final reported energy
        final_energy_kwh = cc_energy_kwh if (cc_energy_kwh > 0) else energy_kwh
        final_energy_joules = final_energy_kwh * (3600.0 * 1000.0)
        final_co2_g = (cc_emissions_kg * 1000.0) if (cc_emissions_kg > 0) else co2_g

        # System total memory
        total_sys_ram_mb = psutil.virtual_memory().total / (1024 * 1024)

        return {
            "duration_s": round(duration_s, 5),
            "avg_cpu_pct": round(avg_cpu_pct, 2),
            "avg_proc_cpu_pct": round(avg_proc_cpu_pct, 2),
            "peak_cpu_pct": round(peak_cpu_pct, 2),
            "baseline_ram_mb": round(self._baseline_rss_mb, 2),
            "peak_ram_mb": round(peak_mem_mb, 2),
            "delta_ram_mb": round(delta_mem_mb, 2),
            "system_ram_total_mb": round(total_sys_ram_mb, 2),
            "energy_joules": round(final_energy_joules, 4),
            "energy_kwh": float(f"{final_energy_kwh:.8f}"),
            "co2_emissions_g": float(f"{final_co2_g:.6f}"),
            "power_model_joules": round(energy_joules, 4),
            # Metadata & methodology transparency
            "is_energy_estimated": True,
            "power_measurement_method": "TDP-calibrated empirical integration (CPU + DRAM)",
            "gpu_supported": False,
            "gpu_monitored": False,
            "tdp_watts": self.power_model.tdp_watts,
            "carbon_intensity_factor_g_per_kwh": self.power_model.carbon_intensity_g_per_kwh
        }
