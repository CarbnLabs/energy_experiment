"""Power monitoring via baseline-subtraction method.

Auto-detects platform:
- macOS Apple Silicon: powermetrics (CPU+GPU+ANE)
- NVIDIA GPU (Linux/Windows): pynvml (hardware energy counter or power sampling)
"""

import time
import subprocess
import threading
import re
import os
import signal
import platform

import numpy as np
import psutil

# ─── GPU/chip detection at import time ───

# Detect macOS Apple Silicon chip
_MAC_CHIP_NAME = None
_MAC_GPU_CORES = 0
if platform.system() == 'Darwin':
    try:
        _MAC_CHIP_NAME = subprocess.check_output(
            ['sysctl', '-n', 'machdep.cpu.brand_string'],
            text=True, timeout=3
        ).strip()
    except Exception:
        pass
    try:
        sp = subprocess.check_output(
            ['system_profiler', 'SPDisplaysDataType'],
            text=True, timeout=5
        )
        m = re.search(r'Total Number of Cores:\s*(\d+)', sp)
        if m:
            _MAC_GPU_CORES = int(m.group(1))
    except Exception:
        pass

# Detect NVIDIA GPUs via pynvml
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
    _GPU_COUNT = pynvml.nvmlDeviceGetCount()
    _GPU_NAME = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
    try:
        pynvml.nvmlDeviceGetTotalEnergyConsumption(
            pynvml.nvmlDeviceGetHandleByIndex(0)
        )
        _NVML_HAS_ENERGY = True
    except pynvml.NVMLError:
        _NVML_HAS_ENERGY = False
except Exception:
    _NVML_AVAILABLE = False
    _NVML_HAS_ENERGY = False
    _GPU_COUNT = 0
    _GPU_NAME = None


def gpu_info():
    """Return dict with GPU/chip detection results."""
    if _NVML_AVAILABLE:
        return {
            'device_type': 'nvidia_gpu',
            'gpu_name': _GPU_NAME,
            'gpu_count': _GPU_COUNT,
            'gpu_cores': None,
            'nvml_available': True,
            'nvml_has_energy': _NVML_HAS_ENERGY,
        }
    if _MAC_CHIP_NAME:
        return {
            'device_type': 'apple_silicon',
            'gpu_name': _MAC_CHIP_NAME,
            'gpu_count': 1,
            'gpu_cores': _MAC_GPU_CORES,
            'nvml_available': False,
            'nvml_has_energy': False,
        }
    return {
        'device_type': 'cpu_only',
        'gpu_name': None,
        'gpu_count': 0,
        'gpu_cores': None,
        'nvml_available': False,
        'nvml_has_energy': False,
    }


class PowerMonitor:
    """Monitors power using baseline-subtraction method.

    Auto-detects platform:
    - macOS Apple Silicon: powermetrics (CPU+GPU+ANE power)
    - NVIDIA GPU (Linux/Windows): pynvml (hardware energy counter or sampling)

    Workflow:
    1. measure_baseline() -- sample system/GPU power for a few seconds while idle
    2. start() -- begin recording power + CPU% during inference
    3. stop() -- end recording, subtract baseline to isolate inference energy

    CPU% is tracked as a sanity check to flag if unexpected processes spike.
    """

    BASELINE_DURATION_S = 3
    CPU_SPIKE_THRESHOLD = 15.0

    def __init__(self, sample_interval_ms=200, process_name='ollama'):
        self.sample_interval_ms = sample_interval_ms
        self.process_name = process_name
        self.power_samples = []
        self.cpu_samples = []
        self.baseline_power_w = None
        self.baseline_cpu = None
        self._process = None
        self._reader_thread = None
        self._cpu_thread = None
        self._gpu_thread = None
        self._running = False
        self._ollama_pids = []
        self._nvml_energy_start = None
        self._nvml_energy_end = None
        self._device_info = gpu_info()
        self._backend = self._detect_backend()
        print(f'PowerMonitor: backend={self._backend} (baseline-subtraction mode)')

    def _detect_backend(self):
        if _NVML_AVAILABLE:
            return 'nvml_energy' if _NVML_HAS_ENERGY else 'nvml_sample'
        if platform.system() == 'Darwin' and self._check_powermetrics():
            return 'powermetrics'
        return 'time_estimate'

    def _check_powermetrics(self):
        try:
            result = subprocess.run(
                ['sudo', '-n', 'powermetrics', '--samplers', 'cpu_power,gpu_power',
                 '-i', '100', '-n', '1'],
                capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    def _find_ollama_pids(self):
        pids = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                name = proc.info['name'].lower()
                cmdline = ' '.join(proc.info.get('cmdline') or []).lower()
                if self.process_name in name or self.process_name in cmdline:
                    pids.append(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return pids

    # ─── NVIDIA helpers ───

    def _nvml_get_total_energy_mj(self):
        total = 0
        for i in range(_GPU_COUNT):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            total += pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        return total

    def _nvml_get_power_w(self):
        total_mw = 0
        for i in range(_GPU_COUNT):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            total_mw += pynvml.nvmlDeviceGetPowerUsage(handle)
        return total_mw / 1000.0

    def _nvml_sample_loop(self):
        interval = self.sample_interval_ms / 1000.0
        while self._running:
            try:
                watts = self._nvml_get_power_w()
                self.power_samples.append((time.time(), watts))
            except Exception:
                pass
            time.sleep(interval)

    # ─── macOS powermetrics helpers ───

    def _read_powermetrics_samples(self, duration_s=None):
        samples = []
        proc = subprocess.Popen(
            ['sudo', '-n', 'powermetrics', '--samplers', 'cpu_power,gpu_power',
             '-i', str(self.sample_interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, preexec_fn=os.setsid)
        start = time.time()
        accumulated = {}
        try:
            for line in iter(proc.stdout.readline, ''):
                if duration_s is not None and (time.time() - start) >= duration_s:
                    break
                if duration_s is None and not self._running:
                    break
                line = line.strip()
                for label in ['CPU Power', 'GPU Power', 'ANE Power']:
                    m = re.search(rf'{label}.*?:\s*(\d+)\s*mW', line)
                    if m:
                        accumulated[label] = float(m.group(1)) / 1000.0
                if '****' in line and accumulated:
                    samples.append((time.time(), sum(accumulated.values())))
                    accumulated = {}
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        return samples

    def _powermetrics_reader_loop(self):
        accumulated = {}
        for line in iter(self._process.stdout.readline, ''):
            if not self._running:
                break
            line = line.strip()
            for label in ['CPU Power', 'GPU Power', 'ANE Power']:
                m = re.search(rf'{label}.*?:\s*(\d+)\s*mW', line)
                if m:
                    accumulated[label] = float(m.group(1)) / 1000.0
            if '****' in line and accumulated:
                self.power_samples.append((time.time(), sum(accumulated.values())))
                accumulated = {}

    # ─── CPU monitoring (shared) ───

    def _cpu_monitor_loop(self):
        psutil.cpu_percent(interval=None)
        procs = {}
        for pid in self._ollama_pids:
            try:
                p = psutil.Process(pid)
                p.cpu_percent(interval=None)
                procs[pid] = p
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        interval = self.sample_interval_ms / 1000.0
        while self._running:
            time.sleep(interval)
            if not self._running:
                break
            total_cpu = psutil.cpu_percent(interval=None)
            ollama_cpu = 0.0
            dead_pids = []
            for pid, p in procs.items():
                try:
                    ollama_cpu += p.cpu_percent(interval=None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    dead_pids.append(pid)
            for pid in dead_pids:
                del procs[pid]
            current_pids = set(procs.keys())
            new_pids = self._find_ollama_pids()
            for pid in new_pids:
                if pid not in current_pids:
                    try:
                        p = psutil.Process(pid)
                        p.cpu_percent(interval=None)
                        procs[pid] = p
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            self.cpu_samples.append((time.time(), ollama_cpu, total_cpu))

    # ─── Public API ───

    def measure_baseline(self):
        """Measure baseline power while idle. Call BEFORE inference."""
        self._ollama_pids = self._find_ollama_pids()
        psutil.cpu_percent(interval=None)
        baseline_cpu_procs = {}
        for pid in self._ollama_pids:
            try:
                p = psutil.Process(pid)
                p.cpu_percent(interval=None)
                baseline_cpu_procs[pid] = p
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        if self._backend in ('nvml_energy', 'nvml_sample'):
            samples = []
            t0 = time.time()
            while (time.time() - t0) < self.BASELINE_DURATION_S:
                samples.append(self._nvml_get_power_w())
                time.sleep(self.sample_interval_ms / 1000.0)
            self.baseline_power_w = np.mean(samples) if samples else 0.0
        elif self._backend == 'powermetrics':
            baseline_samples = self._read_powermetrics_samples(
                duration_s=self.BASELINE_DURATION_S
            )
            self.baseline_power_w = (
                np.mean([s[1] for s in baseline_samples]) if baseline_samples else 0.0
            )
        else:
            self.baseline_power_w = 0.0

        time.sleep(0.1)
        baseline_total_cpu = psutil.cpu_percent(interval=None)
        baseline_ollama_cpu = 0.0
        for pid, p in baseline_cpu_procs.items():
            try:
                baseline_ollama_cpu += p.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self.baseline_cpu = (baseline_ollama_cpu, baseline_total_cpu)

    def start(self):
        """Start monitoring power during inference."""
        self.power_samples = []
        self.cpu_samples = []
        self._start_time = time.time()
        self._running = True
        self._ollama_pids = self._find_ollama_pids()

        if self._backend == 'nvml_energy':
            self._nvml_energy_start = self._nvml_get_total_energy_mj()
            self._gpu_thread = threading.Thread(
                target=self._nvml_sample_loop, daemon=True
            )
            self._gpu_thread.start()
        elif self._backend == 'nvml_sample':
            self._gpu_thread = threading.Thread(
                target=self._nvml_sample_loop, daemon=True
            )
            self._gpu_thread.start()
        elif self._backend == 'powermetrics':
            self._process = subprocess.Popen(
                ['sudo', '-n', 'powermetrics', '--samplers', 'cpu_power,gpu_power',
                 '-i', str(self.sample_interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, preexec_fn=os.setsid)
            self._reader_thread = threading.Thread(
                target=self._powermetrics_reader_loop, daemon=True
            )
            self._reader_thread.start()

        self._cpu_thread = threading.Thread(
            target=self._cpu_monitor_loop, daemon=True
        )
        self._cpu_thread.start()

    def stop(self):
        """Stop monitoring and compute baseline-subtracted results."""
        self._running = False
        elapsed = time.time() - self._start_time

        if self._backend == 'nvml_energy':
            self._nvml_energy_end = self._nvml_get_total_energy_mj()
            if self._gpu_thread:
                self._gpu_thread.join(timeout=2)
        elif self._backend == 'nvml_sample':
            if self._gpu_thread:
                self._gpu_thread.join(timeout=2)
        elif self._backend == 'powermetrics' and self._process:
            try:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                self._process.wait(timeout=3)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            if self._reader_thread:
                self._reader_thread.join(timeout=2)

        if self._cpu_thread:
            self._cpu_thread.join(timeout=2)

        # System-level power
        gpu_avg_w = None
        gpu_energy_mwh = None
        if self._backend == 'nvml_energy' and self._nvml_energy_start is not None:
            energy_delta_mj = self._nvml_energy_end - self._nvml_energy_start
            sys_energy_j = energy_delta_mj / 1000.0
            sys_energy_mwh = sys_energy_j / 3.6
            sys_avg_w = sys_energy_j / max(elapsed, 0.001)
            gpu_avg_w = sys_avg_w
            gpu_energy_mwh = sys_energy_mwh
        elif self.power_samples:
            watts = [s[1] for s in self.power_samples]
            times = [s[0] for s in self.power_samples]
            sys_energy_j = np.trapezoid(watts, times)
            sys_avg_w = np.mean(watts)
            sys_energy_mwh = sys_energy_j / 3600.0 * 1000
            if self._backend == 'nvml_sample':
                gpu_avg_w = sys_avg_w
                gpu_energy_mwh = sys_energy_mwh
        else:
            sys_avg_w = 18.0
            sys_energy_mwh = sys_avg_w * elapsed / 3600.0 * 1000

        # Baseline subtraction
        baseline_w = self.baseline_power_w if self.baseline_power_w is not None else 0.0
        inference_avg_w = max(sys_avg_w - baseline_w, 0.0)
        inference_energy_mwh = inference_avg_w * elapsed / 3600.0 * 1000
        baseline_gpu_w = None
        gpu_inference_avg_w = None
        gpu_inference_energy_mwh = None

        if self._backend == 'nvml_energy' and self._nvml_energy_start is not None:
            baseline_energy_mwh = baseline_w * elapsed / 3600.0 * 1000
            inference_energy_mwh = max(sys_energy_mwh - baseline_energy_mwh, 0.0)
        if self._backend in ('nvml_energy', 'nvml_sample'):
            baseline_gpu_w = baseline_w
            gpu_inference_avg_w = inference_avg_w
            gpu_inference_energy_mwh = inference_energy_mwh
        elif self._backend == 'powermetrics':
            # powermetrics mixes CPU/GPU/ANE into one stream in this project.
            baseline_gpu_w = None
            gpu_inference_avg_w = None
            gpu_inference_energy_mwh = None

        # CPU sanity check
        cpu_flag = False
        avg_ollama_cpu = 0.0
        avg_total_cpu = 0.0
        avg_other_cpu = 0.0
        baseline_other_cpu = 0.0

        if self.cpu_samples:
            avg_ollama_cpu = np.mean([s[1] for s in self.cpu_samples])
            avg_total_cpu = np.mean([s[2] for s in self.cpu_samples])
            avg_other_cpu = max(avg_total_cpu - avg_ollama_cpu, 0.0)
            if self.baseline_cpu is not None:
                baseline_other_cpu = max(
                    self.baseline_cpu[1] - self.baseline_cpu[0], 0.0
                )
            other_cpu_increase = avg_other_cpu - baseline_other_cpu
            if other_cpu_increase > self.CPU_SPIKE_THRESHOLD:
                cpu_flag = True

        return {
            'elapsed_s': elapsed,
            'sys_avg_power_w': sys_avg_w,
            'sys_energy_mwh': sys_energy_mwh,
            'baseline_power_w': baseline_w,
            'inference_avg_power_w': inference_avg_w,
            'inference_energy_mwh': inference_energy_mwh,
            'energy_wh': inference_energy_mwh / 1000.0,
            'avg_ollama_cpu': avg_ollama_cpu,
            'avg_total_cpu': avg_total_cpu,
            'avg_other_cpu': avg_other_cpu,
            'baseline_other_cpu': baseline_other_cpu,
            'cpu_flag': cpu_flag,
            'n_samples': len(self.power_samples),
            'n_cpu_samples': len(self.cpu_samples),
            'method': self._backend,
            'device_type': self._device_info['device_type'],
            'gpu_telemetry_available': self._backend in ('nvml_energy', 'nvml_sample'),
            'gpu_avg_power_w': gpu_avg_w,
            'gpu_energy_mwh': gpu_energy_mwh,
            'baseline_gpu_power_w': baseline_gpu_w,
            'gpu_inference_avg_power_w': gpu_inference_avg_w,
            'gpu_inference_energy_mwh': gpu_inference_energy_mwh,
            'gpu_name': self._device_info['gpu_name'],
            'gpu_count': self._device_info['gpu_count'],
            'gpu_cores': self._device_info['gpu_cores'],
        }
