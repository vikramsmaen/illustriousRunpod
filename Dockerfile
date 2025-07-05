ARG BASE_IMAGE=nvidia/cuda:11.8-devel-ubuntu22.04
FROM ${BASE_IMAGE} as dev-base

ARG MODEL_URL
ARG BASE_MODEL=runwayml/stable-diffusion-v1-5
ENV MODEL_URL=${MODEL_URL}
ENV BASE_MODEL=${BASE_MODEL}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND noninteractive\
    SHELL=/bin/bash

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt install --yes --no-install-recommends\
    wget\
    bash\
    openssh-server \
    software-properties-common &&\
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install python3.10 python3-pip -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

WORKDIR /opt/ckpt

COPY requirements.txt /opt/ckpt/requirements.txt
RUN pip3 install -r /opt/ckpt/requirements.txt

COPY . /opt/ckpt

# Download the safetensors model if URL provided
RUN if [ ! -z "$MODEL_URL" ]; then \
        echo "Downloading model from: $MODEL_URL"; \
        python3 model_fetcher.py --model_url="$MODEL_URL"; \
    else \
        echo "No MODEL_URL provided, skipping model download"; \
    fi

CMD python3 -u /opt/ckpt/runpod_infer.py --model_url="$MODEL_URL" --base_model="$BASE_MODEL"
