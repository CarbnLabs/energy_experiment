#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
nohup ollama serve > logs/ollama.log 2>&1 &
echo "Ollama server started. Log: logs/ollama.log"

