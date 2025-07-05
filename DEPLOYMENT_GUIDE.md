# 🚀 CivitAI Model Deployment Guide

test

## 📋 Prerequisites

- Your 6.5GB safetensors model file from CivitAI
- Cloud storage service (Google Drive, Dropbox, etc.)
- RunPod account
- Docker installed (for local testing)

## 🔧 Setup Steps

### 1. Upload Your Model to Cloud Storage

**Option A: Google Drive**

1. Upload your `.safetensors` file to Google Drive
2. Right-click → Share → Anyone with the link can view
3. Get the share link: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`
4. Convert to direct download: `https://drive.google.com/uc?export=download&id=FILE_ID`

**Option B: Dropbox**

1. Upload your `.safetensors` file to Dropbox
2. Right-click → Share → Create link
3. Change `?dl=0` to `?dl=1` at the end of the URL

**Option C: Hugging Face Hub**

1. Create a new model repository on Hugging Face
2. Upload your `.safetensors` file
3. Get direct URL: `https://huggingface.co/YOUR_USERNAME/YOUR_MODEL/resolve/main/model.safetensors`

### 2. Configure Your Model

Edit `model_config.json`:

```json
{
  "model_info": {
    "name": "Your Model Name",
    "base_model": "runwayml/stable-diffusion-v1-5",
    "model_url": "YOUR_DIRECT_DOWNLOAD_URL_HERE"
  }
}
```

**Determine Base Model Type:**

- If your CivitAI model is based on SD 1.5: `"runwayml/stable-diffusion-v1-5"`
- If based on SD 2.1: `"stabilityai/stable-diffusion-2-1"`
- If based on SDXL: `"stabilityai/stable-diffusion-xl-base-1.0"`

### 3. Build and Test Locally (Optional)

```bash
# Test the download
python model_fetcher.py --model_url="YOUR_DIRECT_DOWNLOAD_URL"

# Build Docker image
docker build --build-arg MODEL_URL="YOUR_DIRECT_DOWNLOAD_URL" --build-arg BASE_MODEL="runwayml/stable-diffusion-v1-5" -t your-civitai-model:latest .

# Test locally
docker run --gpus all -p 8000:8000 your-civitai-model:latest
```

### 4. Deploy to RunPod

**Method 1: GitHub Repo (Recommended)**

1. Push this project to your GitHub repository
2. In RunPod Templates, create a new template:
   - **Template Name**: Your Model Name
   - **Template Type**: Serverless
   - **Container Image**: `runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04`
   - **Docker Build Arguments**:
     ```
     MODEL_URL=YOUR_DIRECT_DOWNLOAD_URL
     BASE_MODEL=runwayml/stable-diffusion-v1-5
     ```
   - **Container Registry Credentials**: Your Docker Hub credentials
   - **GitHub Repository**: Your repository URL

**Method 2: Pre-built Docker Image**

1. Build and push to Docker Hub:
   ```bash
   docker build --build-arg MODEL_URL="YOUR_URL" -t your-dockerhub/model:latest .
   docker push your-dockerhub/model:latest
   ```
2. Use the pushed image directly in RunPod template

### 5. RunPod Template Configuration

**Template Settings:**

- **Container Disk**: 20GB (for model + dependencies)
- **Volume Disk**: 5GB (for temporary files)
- **GPU**: RTX 3090/4090 or A100 (recommended for 6.5GB model)
- **Environment Variables**:
  ```
  MODEL_URL=YOUR_DIRECT_DOWNLOAD_URL
  BASE_MODEL=runwayml/stable-diffusion-v1-5
  ```

### 6. API Usage

**Endpoint URL**: Your RunPod endpoint URL

**Request Format**:

```json
{
  "input": {
    "prompt": "a beautiful landscape, highly detailed",
    "negative_prompt": "blurry, low quality",
    "width": 512,
    "height": 512,
    "num_outputs": 1,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "scheduler": "DPMSolverMultistep",
    "seed": 42
  }
}
```

## 🎯 Model-Specific Notes

### For SD 1.5 Based Models:

- Use `"runwayml/stable-diffusion-v1-5"` as base
- Recommended resolution: 512x512
- Works with all schedulers

### For SD 2.1 Based Models:

- Use `"stabilityai/stable-diffusion-2-1"` as base
- Recommended resolution: 768x768
- Better with DPMSolver scheduler

### For SDXL Based Models:

- Use `"stabilityai/stable-diffusion-xl-base-1.0"` as base
- Recommended resolution: 1024x1024
- Requires more VRAM (24GB+ recommended)

## 🚨 Troubleshooting

**Model Download Fails:**

- Verify your direct download URL works in browser
- Check file size matches expected (6.5GB)
- Ensure URL doesn't require authentication

**Out of Memory:**

- Use smaller resolution
- Reduce num_outputs
- Use gradient checkpointing

**Poor Results:**

- Verify correct base model type
- Check if model requires specific prompt format
- Adjust guidance_scale and steps

## 💰 Cost Optimization

- Use spot instances for development
- Monitor usage with RunPod dashboard
- Consider caching strategies for frequently used models

---

## 📞 Support

If you encounter issues:

1. Check RunPod logs for error messages
2. Test locally with Docker first
3. Verify model compatibility with base model
