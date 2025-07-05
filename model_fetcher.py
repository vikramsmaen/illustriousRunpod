'''
RunPod | serverless-ckpt-template | model_fetcher.py

Downloads the safetensors model from a direct URL.
'''

import os
import shutil
import requests
import argparse
from pathlib import Path
from urllib.parse import urlparse

SAFETENSORS_CACHE_DIR = "safetensors-cache"


def download_safetensors_model(model_url: str):
    '''
    Downloads the safetensors model from the direct URL.
    '''
    model_cache_path = Path(SAFETENSORS_CACHE_DIR)
    if model_cache_path.exists():
        shutil.rmtree(model_cache_path)
    model_cache_path.mkdir(parents=True, exist_ok=True)

    model_filename = "model.safetensors"
    model_path = model_cache_path / model_filename

    print(f"Downloading safetensors model from {model_url}")
    print("This may take several minutes for large models...")
    
    # Download with progress indication
    response = requests.get(model_url, stream=True, timeout=600)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded_size = 0
    
    with open(model_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
            if chunk:
                f.write(chunk)
                downloaded_size += len(chunk)
                if total_size > 0:
                    progress = (downloaded_size / total_size) * 100
                    print(f"Download progress: {progress:.1f}% ({downloaded_size / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)")

    print(f"Safetensors model downloaded to {model_path}")
    return str(model_path)


# ---------------------------------------------------------------------------- #
#                                Parse Arguments                               #
# ---------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--model_url", type=str,
    required=True,
    help="Direct URL to the safetensors model file."
)

if __name__ == "__main__":
    args = parser.parse_args()
    download_safetensors_model(args.model_url)
