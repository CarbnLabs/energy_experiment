"""CSV schema definition — single source of truth for all experiment data."""

from datetime import datetime
from pathlib import Path

import pandas as pd

CSV_SCHEMA = {
    'timestamp':            'ISO 8601 timestamp of the measurement',
    'model':                'Ollama model name, e.g. "llama3.2:3b"',
    'model_family':         'Model family from Ollama API, e.g. "llama"',
    'model_params':         'Parameter count string, e.g. "3B"',
    'model_quant':          'Quantization level, e.g. "Q4_0"',
    'task':                 'Task identifier: "grammar_fix", "translation", "content_generation", "code_generation"',
    'prompt_id':            'Specific prompt ID for generation tasks (e.g. "essay_pollution"), None for text processing',
    'cache_mode':           '"warm" (model cached) or "cold" (model unloaded before run)',
    'target_input_tokens':  'Requested input token count',
    'target_output_tokens': 'Requested max output token count',
    'actual_input_tokens':  'Actual input tokens reported by Ollama',
    'actual_output_tokens': 'Actual output tokens generated',
    'token_method':         '"api" if Ollama reported counts, "estimated" if char-based',
    'response_chars':       'Character count of the generated response',
    'repetition':           'Repetition index (0-based) within the same config',
    'wall_time_s':          'Total wall-clock time for inference in seconds',
    'prompt_eval_s':        'Time spent on prompt evaluation / prefill in seconds',
    'eval_s':               'Time spent on token generation / decode in seconds',
    'load_s':               'Time spent loading model in seconds',
    'tokens_per_sec':       'Output token generation speed (tokens/sec)',
    'sys_avg_power_w':      'Average total system/GPU power during inference (watts)',
    'sys_energy_mwh':       'Total system/GPU energy during inference (milliwatt-hours)',
    'baseline_power_w':     'Baseline idle power measured before inference (watts)',
    'avg_power_w':          'Inference-only power = sys - baseline (watts)',
    'energy_mwh':           'Inference-only energy, baseline-subtracted (milliwatt-hours)',
    'energy_wh':            'Same as energy_mwh / 1000 (watt-hours)',
    'avg_ollama_cpu':       'Average Ollama process CPU% during inference',
    'avg_total_cpu':        'Average total system CPU% during inference',
    'avg_other_cpu':        'Average non-Ollama CPU% during inference',
    'baseline_other_cpu':   'Baseline non-Ollama CPU% before inference',
    'cpu_flag':             'True if non-Ollama CPU spiked >15% above baseline',
    'power_method':         'Backend: "powermetrics", "nvml_energy", "nvml_sample", "time_estimate"',
    'power_samples':        'Number of power samples collected during inference',
    'gpu_name':             'GPU/chip name (e.g. "Apple M5", "NVIDIA RTX 4090")',
    'gpu_count':            'Number of GPUs/chips (1 for Apple Silicon, N for NVIDIA multi-GPU)',
    'gpu_cores':            'GPU core count (Apple Silicon only, None for NVIDIA)',
    # Model architecture metadata
    'architecture':         'Model architecture (e.g. "llama", "gemma2", "qwen2")',
    'parameter_count':      'Total parameter count (exact, from GGUF metadata)',
    'embedding_length':     'Hidden dimension / embedding size',
    'block_count':          'Number of transformer layers',
    'feed_forward_length':  'FFN intermediate dimension',
    'attention_head_count': 'Number of attention heads (Q)',
    'attention_head_count_kv': 'Number of KV heads (for GQA/MQA)',
    'attention_key_length': 'Dimension per attention key head',
    'attention_value_length': 'Dimension per attention value head',
    'attention_type':       'Attention mechanism: "MHA", "GQA", "MQA"',
    'context_length':       'Maximum context window (tokens)',
    'vocab_size':           'Vocabulary size',
    'is_moe':               'True if Mixture of Experts model',
    'expert_count':         'Number of MoE experts (None if dense)',
    'expert_used_count':    'Number of experts activated per token (None if dense)',
    'embedding_params':     'Estimated embedding parameters (embedding_length * vocab_size)',
    'attention_params':     'Estimated total self-attention parameters across all layers',
    'ffn_params':           'Estimated total FFN parameters across all layers',
}

CSV_COLUMNS = list(CSV_SCHEMA.keys())


def build_record(result, power, config):
    """Assemble a measurement record from inference result, power data, and config.

    Args:
        result: Dict from ollama.run_inference().
        power: Dict from PowerMonitor.stop().
        config: Dict with keys from get_model_info() plus: model, task,
                cache_mode, target_input_tokens, target_output_tokens, repetition.

    Returns:
        Dict with all CSV_COLUMNS populated.
    """
    return {
        'timestamp': datetime.now().isoformat(),
        'model': config['model'],
        'model_family': config.get('model_family', config.get('family', 'unknown')),
        'model_params': config.get('model_params', config.get('parameter_size', 'unknown')),
        'model_quant': config.get('model_quant', config.get('quantization_level', 'unknown')),
        'task': config['task'],
        'prompt_id': config.get('prompt_id'),
        'cache_mode': config['cache_mode'],
        'target_input_tokens': config['target_input_tokens'],
        'target_output_tokens': config['target_output_tokens'],
        'actual_input_tokens': result['input_tokens'],
        'actual_output_tokens': result['output_tokens'],
        'token_method': result['token_method'],
        'response_chars': result['response_chars'],
        'repetition': config.get('repetition', 0),
        'wall_time_s': result['wall_time_s'],
        'prompt_eval_s': result['prompt_eval_ns'] / 1e9,
        'eval_s': result['eval_ns'] / 1e9,
        'load_s': result['load_ns'] / 1e9,
        'tokens_per_sec': result['tokens_per_sec'],
        'sys_avg_power_w': power['sys_avg_power_w'],
        'sys_energy_mwh': power['sys_energy_mwh'],
        'baseline_power_w': power['baseline_power_w'],
        'avg_power_w': power['inference_avg_power_w'],
        'energy_mwh': power['inference_energy_mwh'],
        'energy_wh': power['energy_wh'],
        'avg_ollama_cpu': power['avg_ollama_cpu'],
        'avg_total_cpu': power['avg_total_cpu'],
        'avg_other_cpu': power['avg_other_cpu'],
        'baseline_other_cpu': power['baseline_other_cpu'],
        'cpu_flag': power['cpu_flag'],
        'power_method': power['method'],
        'power_samples': power['n_samples'],
        'gpu_name': power.get('gpu_name'),
        'gpu_count': power.get('gpu_count', 0),
        'gpu_cores': power.get('gpu_cores'),
        # Model architecture metadata
        'architecture': config.get('architecture'),
        'parameter_count': config.get('parameter_count'),
        'embedding_length': config.get('embedding_length'),
        'block_count': config.get('block_count'),
        'feed_forward_length': config.get('feed_forward_length'),
        'attention_head_count': config.get('attention_head_count'),
        'attention_head_count_kv': config.get('attention_head_count_kv'),
        'attention_key_length': config.get('attention_key_length'),
        'attention_value_length': config.get('attention_value_length'),
        'attention_type': config.get('attention_type'),
        'context_length': config.get('context_length'),
        'vocab_size': config.get('vocab_size'),
        'is_moe': config.get('is_moe'),
        'expert_count': config.get('expert_count'),
        'expert_used_count': config.get('expert_used_count'),
        'embedding_params': config.get('embedding_params'),
        'attention_params': config.get('attention_params'),
        'ffn_params': config.get('ffn_params'),
    }


def validate_record(record):
    """Check that a record has all required columns."""
    missing = [c for c in CSV_COLUMNS if c not in record]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    return True


def load_results(path):
    """Load a results CSV, enforcing column presence."""
    df = pd.read_csv(path)
    return df


def save_results(measurements, path):
    """Save measurements list to CSV with consistent column order."""
    df = pd.DataFrame(measurements)
    # Reorder to schema order, adding missing columns as NaN
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df[CSV_COLUMNS].to_csv(path, index=False)
    return df


def print_schema():
    """Pretty-print the CSV schema documentation."""
    print(f'{"Column":<25} Description')
    print(f'{"=" * 25} {"=" * 55}')
    for col, desc in CSV_SCHEMA.items():
        print(f'{col:<25} {desc}')
