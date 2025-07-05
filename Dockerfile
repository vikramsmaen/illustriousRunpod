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
echo "🚀 Starting Illustrious Realism deployment..."\n\
echo "📦 Model URL: $MODEL_URL"\n\
echo "🎯 Base Model: $BASE_MODEL"\n\
\n\
# Download model if URL provided\n\
if [ ! -z "$MODEL_URL" ]; then\n\
    echo "⬇️ Downloading model..."\n\
    python model_fetcher.py --model_url="$MODEL_URL"\n\
else\n\
    echo "⚠️ No MODEL_URL provided, skipping model download"\n\
fi\n\
\n\
# Start the inference server\n\
echo "🔥 Starting inference server..."\n\
python -u runpod_infer.py --model_url="$MODEL_URL" --base_model="$BASE_MODEL"\n\
' > /start.sh && chmod +x /start.sh

# RunPod serverless handler
CMD ["/start.sh"]
