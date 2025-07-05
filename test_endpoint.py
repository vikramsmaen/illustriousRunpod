#!/usr/bin/env python3
"""
Test script for local development and API endpoint testing
"""

import requests
import json
import argparse
import time

def test_local_endpoint(url="http://localhost:8000", prompt="a beautiful landscape"):
    """Test local Docker container"""
    payload = {
        "input": {
            "prompt": prompt,
            "negative_prompt": "blurry, low quality, bad anatomy",
            "width": 512,
            "height": 512,
            "num_outputs": 1,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "scheduler": "DPMSolverMultistep",
            "seed": 42
        }
    }
    
    print(f"Testing endpoint: {url}")
    print(f"Prompt: {prompt}")
    
    start_time = time.time()
    
    try:
        response = requests.post(f"{url}/runsync", json=payload, timeout=300)
        end_time = time.time()
        
        print(f"Response time: {end_time - start_time:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(json.dumps(result, indent=2))
        else:
            print("❌ Error!")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_runpod_endpoint(endpoint_id, api_key, prompt="a beautiful landscape"):
    """Test RunPod serverless endpoint"""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "input": {
            "prompt": prompt,
            "negative_prompt": "blurry, low quality, bad anatomy",
            "width": 512,
            "height": 512,
            "num_outputs": 1,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "scheduler": "DPMSolverMultistep",
            "seed": 42
        }
    }
    
    print(f"Testing RunPod endpoint: {endpoint_id}")
    print(f"Prompt: {prompt}")
    
    start_time = time.time()
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=300)
        end_time = time.time()
        
        print(f"Response time: {end_time - start_time:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            print(json.dumps(result, indent=2))
        else:
            print("❌ Error!")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the image generation endpoint")
    parser.add_argument("--mode", choices=["local", "runpod"], default="local", help="Test mode")
    parser.add_argument("--url", default="http://localhost:8000", help="Local endpoint URL")
    parser.add_argument("--endpoint-id", help="RunPod endpoint ID")
    parser.add_argument("--api-key", help="RunPod API key")
    parser.add_argument("--prompt", default="a beautiful landscape, highly detailed, masterpiece", help="Test prompt")
    
    args = parser.parse_args()
    
    if args.mode == "local":
        test_local_endpoint(args.url, args.prompt)
    elif args.mode == "runpod":
        if not args.endpoint_id or not args.api_key:
            print("❌ RunPod endpoint ID and API key required for RunPod mode")
            exit(1)
        test_runpod_endpoint(args.endpoint_id, args.api_key, args.prompt)
