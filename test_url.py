import requests
import os

# Replace this with your actual Hugging Face URL
model_url = "https://huggingface.co/datasets/YOUR_USERNAME/YOUR_DATASET_NAME/resolve/main/illustriousRealism_v10VAE.safetensors"

print("🔗 Testing Hugging Face URL...")
print(f"URL: {model_url}")

try:
    # Test if URL is accessible
    response = requests.head(model_url, timeout=30)
    print(f"📊 Status Code: {response.status_code}")
    
    if response.status_code == 200:
        file_size = response.headers.get('content-length')
        if file_size:
            size_gb = int(file_size) / (1024**3)
            print(f"📏 File Size: {size_gb:.2f} GB")
        print("✅ URL is accessible! Ready for deployment.")
    else:
        print(f"❌ URL not accessible. Status: {response.status_code}")
        print("💡 Make sure your dataset is public or check the URL format")
        
except Exception as e:
    print(f"❌ Error testing URL: {e}")
    print("💡 Please check your internet connection and URL format")

print("\n📋 Next steps:")
print("1. Update the model_url variable above with your real URL")
print("2. Run this script again to verify")
print("3. Proceed to RunPod deployment")
