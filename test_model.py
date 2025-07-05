import os
from safetensors.torch import load_file

# Update this to match your actual model filename
model_path = 'models/illustriousRealism_v10VAE.safetensors'

if os.path.exists(model_path):
    print(f'✅ Model found: {model_path}')
    file_size_gb = os.path.getsize(model_path) / (1024*1024*1024)
    print(f'📏 File size: {file_size_gb:.2f} GB')
    
    try:
        print('🔄 Loading model...')
        state_dict = load_file(model_path)
        print(f'🔑 Keys found: {len(state_dict)} parameters')
        
        # Check for common key patterns
        unet_keys = [k for k in state_dict.keys() if 'model.diffusion_model' in k or 'unet' in k.lower()]
        vae_keys = [k for k in state_dict.keys() if 'first_stage_model' in k or 'vae' in k.lower()]
        text_keys = [k for k in state_dict.keys() if 'cond_stage_model' in k or 'text' in k.lower()]
        
        print(f'🎯 UNet keys: {len(unet_keys)}')
        print(f'🎨 VAE keys: {len(vae_keys)}')
        print(f'📝 Text encoder keys: {len(text_keys)}')
        
        if unet_keys:
            print('✅ This appears to be a valid Stable Diffusion model!')
        else:
            print('⚠️ No UNet keys found - this might not be a standard SD model')
            
        print('✅ Model loads successfully!')
        
    except Exception as e:
        print(f'❌ Error loading model: {e}')
else:
    print(f'❌ Model file not found at: {model_path}')
    print('📁 Files in models directory:')
    if os.path.exists('models'):
        for file in os.listdir('models'):
            print(f'   - {file}')
    else:
        print('   models/ directory not found!')
