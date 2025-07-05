#!/bin/bash

# RunPod startup script
echo "🚀 Starting Illustrious Realism deployment..."

# Set default values if not provided
MODEL_URL=${MODEL_URL:-""}
BASE_MODEL=${BASE_MODEL:-"runwayml/stable-diffusion-v1-5"}

echo "📦 Model URL: $MODEL_URL"
echo "🎯 Base Model: $BASE_MODEL"

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Download model if URL provided
if [ ! -z "$MODEL_URL" ]; then
    echo "⬇️ Downloading model..."
    python model_fetcher.py --model_url="$MODEL_URL"
else
    echo "⚠️ No MODEL_URL provided, skipping model download"
fi

# Start the inference server
echo "🔥 Starting inference server..."
python runpod_infer.py --model_url="$MODEL_URL" --base_model="$BASE_MODEL"
