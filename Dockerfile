FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

ARG MODEL_URL
ARG BASE_MODEL=runwayml/stable-diffusion-v1-5
ENV MODEL_URL=${MODEL_URL}
ENV BASE_MODEL=${BASE_MODEL}

WORKDIR /

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all project files
COPY . .

# Create a startup script that will download model at runtime
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Function for logging with timestamps\n\
log() {\n\
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] $1"\n\
}\n\
\n\
log "🚀 Starting Illustrious Realism deployment..."\n\
log "📦 Model URL: $MODEL_URL"\n\
log "🎯 Base Model: $BASE_MODEL"\n\
log "💾 Available disk space: $(df -h / | tail -1 | awk '"'"'{print $4}'"'"')"\n\
log "🧠 Available memory: $(free -h | grep Mem | awk '"'"'{print $7}'"'"')"\n\
\n\
# Create cache directory\n\
mkdir -p safetensors-cache\n\
log "📁 Created cache directory"\n\
\n\
# Check if we have environment variables\n\
if [ -z "$MODEL_URL" ]; then\n\
    log "❌ MODEL_URL environment variable is not set!"\n\
    exit 1\n\
fi\n\
\n\
# Download model if URL provided and not already cached\n\
if [ ! -f "safetensors-cache/model.safetensors" ]; then\n\
    log "⬇️ Downloading model (this may take 5-10 minutes)..."\n\
    log "🌐 URL: $MODEL_URL"\n\
    \n\
    # Download with timeout and better error handling\n\
    python model_fetcher.py --model_url="$MODEL_URL" 2>&1 | while read line; do\n\
        log "📥 $line"\n\
    done\n\
    \n\
    if [ $? -eq 0 ] && [ -f "safetensors-cache/model.safetensors" ]; then\n\
        log "✅ Model download completed successfully"\n\
        log "📏 Model size: $(du -h safetensors-cache/model.safetensors | cut -f1)"\n\
    else\n\
        log "❌ Model download failed!"\n\
        exit 1\n\
    fi\n\
else\n\
    log "✅ Model already cached"\n\
    log "📏 Cached model size: $(du -h safetensors-cache/model.safetensors | cut -f1)"\n\
fi\n\
\n\
# Check Python dependencies\n\
log "🐍 Checking Python environment..."\n\
python -c "import torch; print(f'\''PyTorch version: {torch.__version__}'\'')" || {\n\
    log "❌ PyTorch import failed!"\n\
    exit 1\n\
}\n\
\n\
python -c "import diffusers; print(f'\''Diffusers version: {diffusers.__version__}'\'')" || {\n\
    log "❌ Diffusers import failed!"\n\
    exit 1\n\
}\n\
\n\
# Start the inference server\n\
log "🔥 Starting inference server..."\n\
exec python -u runpod_infer.py --model_url="$MODEL_URL" --base_model="$BASE_MODEL"\n\
' > /start.sh && chmod +x /start.sh

# RunPod serverless handler
CMD ["/start.sh"]
