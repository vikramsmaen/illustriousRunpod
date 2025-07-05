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

# Download the safetensors model if URL provided
RUN if [ ! -z "$MODEL_URL" ]; then \
        echo "Downloading model from: $MODEL_URL"; \
        python model_fetcher.py --model_url="$MODEL_URL"; \
    else \
        echo "No MODEL_URL provided, skipping model download"; \
    fi

# RunPod serverless handler
CMD python -u runpod_infer.py --model_url="$MODEL_URL" --base_model="$BASE_MODEL"
