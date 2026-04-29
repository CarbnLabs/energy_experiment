# Ollama Energy Experiment

This folder is now focused on a single workflow: run the experiment with `Ollama` and choose where the CSV is stored.

You can pick one output mode in the config:

- `local`: save the CSV inside this project under `data/`
- `s3`: save the CSV directly to an S3 bucket

The experiment logic still reuses the task set and measurement flow from `carbntrace`, but the runner here is self-contained.

## GPU Measurement

On NVIDIA-backed machines such as `g4dn.xlarge`, the project now checks for NVIDIA/NVML telemetry at startup and writes explicit GPU fields into the CSV when available.

New GPU-focused CSV columns include:

- `device_type`
- `gpu_telemetry_available`
- `gpu_avg_power_w`
- `gpu_energy_mwh`
- `baseline_gpu_power_w`
- `gpu_inference_avg_power_w`
- `gpu_inference_energy_mwh`

If the run falls back to `power_method = time_estimate`, then real GPU telemetry was not available and those GPU-specific fields will be empty.

## Folder layout

- `run_experiment.py`: main CLI entry point
- `config.example.json`: Ollama config for local CSV output
- `config.s3.example.json`: Ollama config for S3 CSV output
- `requirements-base.txt`: dependencies for Ollama runs
- `start_ollama_server.sh`: starts Ollama
- `data/`: local CSV output folder when `output_mode` is `local`
- `energy_experiment/`: bundled runner, tasks, power, schema, and backend code

## Setup

```bash
cd energy_experiment
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-base.txt
```

Install and start Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
./start_ollama_server.sh
```

Pull the models you want:

```bash
ollama pull llama3.2:3b
ollama pull gemma2:2b
```

On EC2 GPU instances, verify the NVIDIA device before running:

```bash
nvidia-smi
```

If you want real GPU power in the CSV, make sure the run detects:

- `power_method = nvml_energy` or `nvml_sample`
- `device_type = nvidia_gpu`
- `gpu_telemetry_available = true`

## Option 1: Save Locally

Start from:

```bash
cp config.example.json config.json
```

Important settings:

```json
{
  "provider": "ollama",
  "output_mode": "local",
  "results_file": "data/energy_measurements_ollama.csv"
}
```

Run:

```bash
python run_experiment.py --config config.json
```

Local output:

```bash
data/energy_measurements_ollama.csv
```

The runner resumes from that same local CSV if it already exists.

## Option 2: Save To S3

Create an S3 bucket if needed:

```bash
aws s3 mb s3://YOUR-BUCKET-NAME --region us-east-1
```

Start from:

```bash
cp config.s3.example.json config.json
```

Set:

- `s3_bucket`
- `s3_key`

Important settings:

```json
{
  "provider": "ollama",
  "output_mode": "s3",
  "s3_bucket": "YOUR-BUCKET-NAME",
  "s3_key": "energy-experiments/energy_measurements_ollama.csv"
}
```

Run:

```bash
python run_experiment.py --config config.json
```

S3 output:

```bash
s3://YOUR-BUCKET-NAME/energy-experiments/energy_measurements_ollama.csv
```

In S3 mode, the runner resumes from the CSV already stored in S3 and does not keep a local CSV snapshot.

## Notes

- `output_mode` must be either `local` or `s3`.
- In `local` mode, `results_file` controls the CSV path and defaults to `data/energy_measurements_ollama.csv`.
- In `s3` mode, both `s3_bucket` and `s3_key` are required.
- The same experiment grid, tasks, and power measurement logic are used in both modes.
- On NVIDIA EC2 instances, Ollama may use the GPU automatically, but the CSV only shows explicit GPU energy when NVML telemetry is available on the machine.
