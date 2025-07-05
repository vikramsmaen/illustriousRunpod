# Models Directory

Place your downloaded CivitAI safetensors model file here for local testing.

For production deployment, you'll upload the model to cloud storage and provide the direct download URL.

## File Structure

```
models/
├── your_model.safetensors  # Your CivitAI model (6.5GB)
└── README.md              # This file
```

## Note

- This directory is included in .gitignore to avoid accidentally committing large model files
- For production, use cloud storage URLs instead of local files
