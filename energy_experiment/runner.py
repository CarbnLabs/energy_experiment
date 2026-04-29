"""Experiment runner with configurable local or S3 CSV persistence."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd

from .backends import BackendConfig, build_backend
from .power import PowerMonitor, gpu_info
from .schema import CSV_COLUMNS, build_record, save_results
from .tasks import TASKS, build_prompt, get_task_input


def _build_run_list(
    models: list[str],
    tasks: list[str],
    input_tokens: list[int],
    output_tokens: list[int],
    cache_modes: list[str],
    n_reps: int,
) -> list[dict[str, Any]]:
    run_list: list[dict[str, Any]] = []
    for model_name in models:
        for task in tasks:
            task_info = TASKS.get(task, {})
            task_type = task_info.get("type", "text_processing")

            if task_type == "generation":
                for prompt_entry in task_info.get("prompts", []):
                    for t_out in output_tokens:
                        for cache_mode in cache_modes:
                            for rep in range(n_reps):
                                run_list.append(
                                    {
                                        "model": model_name,
                                        "task": task,
                                        "prompt_id": prompt_entry["id"],
                                        "prompt_text": prompt_entry["text"],
                                        "t_in": 0,
                                        "t_out": t_out,
                                        "cache_mode": cache_mode,
                                        "rep": rep,
                                    }
                                )
            else:
                for t_in in input_tokens:
                    for t_out in output_tokens:
                        for cache_mode in cache_modes:
                            for rep in range(n_reps):
                                run_list.append(
                                    {
                                        "model": model_name,
                                        "task": task,
                                        "prompt_id": None,
                                        "prompt_text": None,
                                        "t_in": t_in,
                                        "t_out": t_out,
                                        "cache_mode": cache_mode,
                                        "rep": rep,
                                    }
                                )
    return run_list


def _upload_to_s3(results_file: Path, bucket: str, key: str) -> None:
    import boto3

    s3 = boto3.client("s3")
    s3.upload_file(str(results_file), bucket, key)


def _serialize_measurements(measurements: list[dict[str, Any]]) -> str:
    df = pd.DataFrame(measurements)
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[CSV_COLUMNS].to_csv(index=False)


def _load_measurements_from_s3(bucket: str, key: str) -> list[dict[str, Any]]:
    import boto3

    s3 = boto3.client("s3")
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except Exception:
        return []

    csv_text = response["Body"].read().decode("utf-8")
    if not csv_text.strip():
        return []
    return pd.read_csv(io.StringIO(csv_text)).to_dict("records")


def _save_measurements_to_s3(
    measurements: list[dict[str, Any]],
    bucket: str,
    key: str,
) -> None:
    import boto3

    s3 = boto3.client("s3")
    csv_text = _serialize_measurements(measurements)
    s3.put_object(Bucket=bucket, Key=key, Body=csv_text.encode("utf-8"))


def _resolve_results_file(config: dict[str, Any], provider: str) -> Path:
    """Resolve the local CSV path inside the project bundle by default."""
    config_dir = Path(config.get("_config_dir", ".")).resolve()
    bundle_dir = Path(config.get("_bundle_dir", config_dir)).resolve()

    configured = config.get("results_file")
    if configured:
        candidate = Path(configured)
        if candidate.is_absolute():
            return candidate
        return (config_dir / candidate).resolve()

    default_name = "energy_measurements.csv"
    if provider == "ollama":
        default_name = "energy_measurements_ollama.csv"
    elif provider == "vllm":
        default_name = "energy_measurements_vllm.csv"

    return (bundle_dir / "data" / default_name).resolve()


def run_experiment(config: dict[str, Any]) -> pd.DataFrame:
    """Run the experiment grid and optionally upload the CSV to S3."""
    backend_config = BackendConfig(
        provider=config.get("provider", "auto"),
        api_base=config.get("api_base", "http://127.0.0.1:11434"),
        temperature=float(config.get("temperature", 0.7)),
        startup_wait_s=int(config.get("startup_wait_s", 30)),
        request_timeout_s=int(config.get("request_timeout_s", 600)),
    )
    backend = build_backend(backend_config)
    print(f"Using backend={backend.provider} api_base={backend_config.api_base}")

    requested_models = config["models"]
    available_models = backend.list_models()
    models = [model for model in requested_models if model in available_models]
    if not models:
        raise RuntimeError(
            f"None of the requested models are available. Requested={requested_models} "
            f"Available={available_models}"
        )

    requested_cache_modes = list(config.get("cache_modes", ["warm", "cold"]))
    if "cold" in requested_cache_modes and not backend.supports_cold_cache():
        print(
            "Cold-cache runs are not supported by the active backend configuration. "
            "Skipping cold runs."
        )
        requested_cache_modes = [mode for mode in requested_cache_modes if mode != "cold"]
    if not requested_cache_modes:
        requested_cache_modes = ["warm"]

    output_mode = config.get("output_mode", "local")
    if output_mode not in {"local", "s3"}:
        raise RuntimeError("output_mode must be either 'local' or 's3'.")

    s3_bucket = config.get("s3_bucket")
    s3_key = config.get("s3_key")
    use_s3 = output_mode == "s3"

    if use_s3 and not (s3_bucket and s3_key):
        raise RuntimeError(
            "S3 output mode requires both s3_bucket and s3_key in the config."
        )

    results_file: Path | None = None
    if use_s3:
        measurements = _load_measurements_from_s3(s3_bucket, s3_key)
        if measurements:
            print(f"Resuming with {len(measurements)} existing measurements from s3://{s3_bucket}/{s3_key}")
        else:
            print(f"No existing S3 CSV found at s3://{s3_bucket}/{s3_key}; starting fresh.")
    else:
        results_file = _resolve_results_file(config, backend.provider)
        if results_file.exists():
            existing = pd.read_csv(results_file)
            measurements = existing.to_dict("records")
            print(f"Resuming with {len(measurements)} existing measurements from {results_file}")
        else:
            measurements = []

    run_list = _build_run_list(
        models=models,
        tasks=config["tasks"],
        input_tokens=config["input_tokens"],
        output_tokens=config["output_tokens"],
        cache_modes=requested_cache_modes,
        n_reps=int(config.get("n_reps", 2)),
    )
    total = len(run_list)
    device_info = gpu_info()
    print(
        "Detected device: type={device_type} gpu_name={gpu_name} "
        "gpu_count={gpu_count} nvml_available={nvml_available} "
        "nvml_has_energy={nvml_has_energy}".format(**device_info)
    )
    if device_info["device_type"] == "nvidia_gpu" and not device_info["nvml_available"]:
        print(
            "Warning: NVIDIA GPU hardware was expected, but NVML telemetry is not "
            "available. GPU-specific CSV fields will be empty."
        )
    monitor = PowerMonitor(process_name=config.get("process_name", backend.provider))

    current_model = None
    for index, run in enumerate(run_list, start=1):
        model_name = run["model"]
        task = run["task"]

        if model_name != current_model:
            current_model = model_name
            print(f"\n{'=' * 70}")
            print(f"MODEL: {model_name}")
            print(f"{'=' * 70}")
            try:
                info = backend.get_model_info(model_name)
                print(
                    "  Architecture: {architecture} | Params: {parameter_count} | "
                    "Family: {family} | Quant: {quantization_level}".format(**info)
                )
            except Exception:
                info = backend.get_model_info(model_name)
                print("  Using fallback model metadata.")

        exists = any(
            measurement.get("model") == model_name
            and measurement.get("task") == task
            and measurement.get("prompt_id") == run["prompt_id"]
            and measurement.get("target_input_tokens") == run["t_in"]
            and measurement.get("target_output_tokens") == run["t_out"]
            and measurement.get("cache_mode") == run["cache_mode"]
            and measurement.get("repetition") == run["rep"]
            for measurement in measurements
        )
        if exists:
            continue

        if run["cache_mode"] == "cold":
            if not backend.cold_reset(model_name):
                print(f"[{index}/{total}] skipped cold run for {model_name} (reset unsupported)")
                continue
        elif run["cache_mode"] == "warm" and run["rep"] == 0:
            backend.warm_model(model_name)

        prompt = run["prompt_text"]
        if prompt is None:
            input_text = get_task_input(task, run["t_in"])
            prompt = build_prompt(task, input_text)

        label = (
            f"[{index}/{total}] {task} in={run['t_in']} out={run['t_out']} "
            f"{run['cache_mode']} rep={run['rep'] + 1}"
        )
        if run["prompt_id"]:
            label = (
                f"[{index}/{total}] {task}:{run['prompt_id']} out={run['t_out']} "
                f"{run['cache_mode']} rep={run['rep'] + 1}"
            )
        print(label, end=" ", flush=True)

        monitor.measure_baseline()
        monitor.start()
        try:
            result = backend.run_inference(model_name, prompt, run["t_out"])
            power = monitor.stop()
        except Exception as exc:
            monitor.stop()
            print(f"FAIL: {exc}")
            continue

        record = build_record(
            result=result,
            power=power,
            config={
                "model": model_name,
                **info,
                "task": task,
                "prompt_id": run["prompt_id"],
                "cache_mode": run["cache_mode"],
                "target_input_tokens": run["t_in"],
                "target_output_tokens": run["t_out"],
                "repetition": run["rep"],
            },
        )
        measurements.append(record)
        print(
            f"-> in:{record['actual_input_tokens']} out:{record['actual_output_tokens']} "
            f"{record['wall_time_s']:.1f}s base:{record['baseline_power_w']:.1f}W "
            f"inf:{record['avg_power_w']:.1f}W {record['energy_mwh']:.1f}mWh "
            f"{record['tokens_per_sec']:.1f}t/s "
            f"gpu:{record.get('gpu_inference_energy_mwh') or 0:.1f}mWh"
        )

        if use_s3:
            _save_measurements_to_s3(measurements, s3_bucket, s3_key)
        else:
            results_file.parent.mkdir(parents=True, exist_ok=True)
            save_results(measurements, results_file)

    if use_s3:
        print(f"\nComplete. {len(measurements)} measurements saved to s3://{s3_bucket}/{s3_key}")
    else:
        print(f"\nComplete. {len(measurements)} measurements saved to {results_file}")
    if use_s3:
        print(f"S3 upload target: s3://{s3_bucket}/{s3_key}")
    return pd.DataFrame(measurements)
