import os
import shutil
import sys

# Ensure safetensors-cache directory exists
os.makedirs('safetensors-cache', exist_ok=True)

# Copy model to expected location
model_source = 'models/illustriousRealism_v10VAE.safetensors'
model_dest = 'safetensors-cache/model.safetensors'

if os.path.exists(model_source):
    print(f'📋 Copying model from {model_source} to {model_dest}')
    shutil.copy2(model_source, model_dest)
    print('✅ Model copied successfully!')
    
    # Test the SD runner
    try:
        import sd_runner
        print('🔄 Initializing predictor...')
        predictor = sd_runner.Predictor(base_model="runwayml/stable-diffusion-v1-5")
        print('🔄 Setting up model (this may take a few minutes)...')
        predictor.setup()
        print('✅ Local model setup completed successfully!')
        
    except Exception as e:
        print(f'❌ Error in setup: {e}')
        print('💡 You may need to install additional dependencies:')
        print('   pip install diffusers transformers accelerate xformers')
        
else:
    print(f'❌ Source model not found: {model_source}')
    print('📁 Available files in models/:')
    if os.path.exists('models'):
        for file in os.listdir('models'):
            print(f'   - {file}')
