import os
import time
from typing import List

import torch
from safetensors.torch import load_file

from diffusers import (
    StableDiffusionPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

# Configuration for safetensors models
SAFETENSORS_CACHE_DIR = "safetensors-cache"
MODEL_FILE = "model.safetensors"


class Predictor:
    ''' A predictor class that loads the safetensors model into memory and runs predictions '''

    def __init__(self, base_model="runwayml/stable-diffusion-v1-5"):
        self.base_model = base_model  # Base model to use (SD 1.5, SD 2.1, or SDXL)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.safetensors_path = os.path.join(SAFETENSORS_CACHE_DIR, MODEL_FILE)

    def setup(self):
        start_time = time.time()
        """Load the base model and apply safetensors weights"""
        print(f"🔄 Loading base pipeline: {self.base_model}")
        
        try:
            # Check available memory
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"🖥️ GPU Memory Available: {gpu_memory:.1f} GB")
            
            # Load base pipeline without safety checker for faster loading
            print("📦 Downloading base model components...")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.base_model,
                safety_checker=None,
                requires_safety_checker=False,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            print(f"🚀 Moving pipeline to {self.device}...")
            self.pipe = self.pipe.to(self.device)

            # Load and apply safetensors weights if available
            if os.path.exists(self.safetensors_path):
                print(f"🔧 Loading custom safetensors weights from {self.safetensors_path}")
                file_size = os.path.getsize(self.safetensors_path) / (1024**3)
                print(f"📏 Safetensors file size: {file_size:.2f} GB")
                
                try:
                    print("📖 Reading safetensors file...")
                    state_dict = load_file(self.safetensors_path)
                    print(f"🔑 Loaded {len(state_dict)} parameter tensors")
                    
                    # Filter and load UNet weights
                    unet_state_dict = {}
                    print("🔍 Filtering UNet weights...")
                    
                    for key, value in state_dict.items():
                        if key.startswith('model.diffusion_model.'):
                            # Handle Automatic1111/CivitAI format
                            new_key = key.replace('model.diffusion_model.', '')
                            unet_state_dict[new_key] = value
                        elif not any(prefix in key for prefix in ['first_stage_model', 'cond_stage_model', 'model.diffusion_model']):
                            # Direct diffusers format
                            unet_state_dict[new_key] = value
                    
                    print(f"🎯 Found {len(unet_state_dict)} UNet parameters")
                    
                    if unet_state_dict:
                        print("⚙️ Loading UNet weights...")
                        missing_keys, unexpected_keys = self.pipe.unet.load_state_dict(unet_state_dict, strict=False)
                        print(f"✅ Successfully loaded custom safetensors weights")
                        if missing_keys:
                            print(f"⚠️ Missing keys: {len(missing_keys)}")
                        if unexpected_keys:
                            print(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
                    else:
                        print("⚠️ No compatible UNet weights found in safetensors file")
                        print("🔍 Available keys preview:", list(state_dict.keys())[:5])
                        
                except Exception as e:
                    print(f"❌ Error loading safetensors: {e}")
                    print("🔄 Continuing with base model...")
            else:
                print(f"⚠️ Safetensors file not found at {self.safetensors_path}")
                print("📦 Using base model only...")

            # Enable memory optimizations
            try:
                print("🚀 Enabling XFormers memory optimization...")
                self.pipe.enable_xformers_memory_efficient_attention()
                print("✅ XFormers enabled")
            except Exception as e:
                print(f"⚠️ XFormers not available: {e}")
                print("🔄 Continuing without XFormers...")
            
            # Set default scheduler
            print("⚙️ Setting up scheduler...")
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            
            # Test a small inference to ensure everything works
            print("🧪 Running test inference...")
            try:
                with torch.inference_mode():
                    test_output = self.pipe(
                        "test", 
                        num_inference_steps=1, 
                        width=64, 
                        height=64,
                        output_type="latent"
                    )
                print("✅ Test inference successful")
            except Exception as e:
                print(f"⚠️ Test inference failed: {e}")
            
            end_time = time.time()
            print(f"🎉 Setup completed successfully in {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Critical error during setup: {e}")
            import traceback
            traceback.print_exc()
            raise e

    @torch.inference_mode()
    def predict(self, prompt, negative_prompt, width, height, num_outputs, num_inference_steps, guidance_scale, scheduler, seed):
        """Run a single prediction on the model"""
        start_time = time.time()
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config)

        generator = torch.Generator(self.device).manual_seed(seed)
        output = self.pipe(
            prompt=[prompt] * num_outputs if prompt is not None else None,
            negative_prompt=[negative_prompt] * num_outputs
            if negative_prompt is not None
            else None,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

        output_paths = []
        for i, sample in enumerate(output.images):
            if output.nsfw_content_detected and output.nsfw_content_detected[i]:
                continue

            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(output_path)

        if len(output_paths) == 0:
            raise Exception(
                "NSFW content detected. Try running it again, or try a different prompt.")
        end_time = time.time()
        print(f"inference took {end_time - start_time} time")
        return output_paths


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]
