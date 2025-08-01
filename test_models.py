#!/usr/bin/env python3
"""
Test script to verify model loading and input shapes
"""

import os
import sys
import tensorflow as tf
import numpy as np
import traceback

def test_model_loading():
    """Test loading of all models and print their input shapes"""
    
    print("Testing model loading...")
    print("=" * 50)
    
    # Test image models
    image_models = [
        "deepfake-detection-tensorflow2-default-v1/image/XceptionModel.keras",
        "deepfake-detection-tensorflow2-default-v1/image/InceptionV3Model.keras",
        "deepfake-detection-tensorflow2-default-v1/image/ResNet50Model.keras",
        "deepfake-detection-tensorflow2-default-v1/image/DenseNet121Model.keras",
        "deepfake-detection-tensorflow2-default-v1/image/deepfake_detection_xception_180k.h5"
    ]
    
    print("Testing Image Models:")
    for model_path in image_models:
        if os.path.exists(model_path):
            try:
                print(f"\nLoading: {model_path}")
                model = tf.keras.models.load_model(model_path, compile=False)
                print(f"✓ Successfully loaded")
                print(f"  Input shape: {model.input_shape}")
                print(f"  Output shape: {model.output_shape}")
                print(f"  Model type: {type(model)}")
            except Exception as e:
                print(f"✗ Failed to load: {e}")
        else:
            print(f"✗ File not found: {model_path}")
    
    print("\n" + "=" * 50)
    
    # Test audio models
    audio_models = [
        "deepfake-detection-tensorflow2-default-v1/audio/audio_classifier.h5",
        "deepfake-detection-tensorflow2-default-v1/audio/audio_classifier_2.h5"
    ]
    
    print("Testing Audio Models:")
    for model_path in audio_models:
        if os.path.exists(model_path):
            try:
                print(f"\nLoading: {model_path}")
                model = tf.keras.models.load_model(model_path, compile=False)
                print(f"✓ Successfully loaded")
                print(f"  Input shape: {model.input_shape}")
                print(f"  Output shape: {model.output_shape}")
                print(f"  Model type: {type(model)}")
            except Exception as e:
                print(f"✗ Failed to load: {e}")
        else:
            print(f"✗ File not found: {model_path}")

def test_input_generation():
    """Test generating sample inputs for each model"""
    
    print("\n" + "=" * 50)
    print("Testing Input Generation:")
    
    # Test image input
    print("\nImage input test:")
    try:
        # Create a sample image input (224x224x3)
        sample_image = np.random.random((1, 224, 224, 3)).astype(np.float32)
        print(f"✓ Generated image input: {sample_image.shape}")
    except Exception as e:
        print(f"✗ Failed to generate image input: {e}")
    
    # Test audio input
    print("\nAudio input test:")
    try:
        # Create a sample audio input (1, 128, 109)
        sample_audio = np.random.random((1, 128, 109)).astype(np.float32)
        print(f"✓ Generated audio input: {sample_audio.shape}")
        
        # Also test 4D input (1, 128, 109, 1)
        sample_audio_4d = np.random.random((1, 128, 109, 1)).astype(np.float32)
        print(f"✓ Generated 4D audio input: {sample_audio_4d.shape}")
    except Exception as e:
        print(f"✗ Failed to generate audio input: {e}")

def test_model_prediction():
    """Test making predictions with loaded models"""
    
    print("\n" + "=" * 50)
    print("Testing Model Predictions:")
    
    # Test with Xception model
    model_path = "deepfake-detection-tensorflow2-default-v1/image/XceptionModel.keras"
    if os.path.exists(model_path):
        try:
            print(f"\nTesting prediction with: {model_path}")
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Generate sample input matching the model's expected shape
            input_shape = model.input_shape[1:]  # Remove batch dimension
            sample_input = np.random.random((1,) + input_shape).astype(np.float32)
            
            print(f"Input shape: {sample_input.shape}")
            prediction = model.predict(sample_input, verbose=0)
            print(f"✓ Prediction successful")
            print(f"  Output shape: {prediction.shape}")
            print(f"  Output values: {prediction[0]}")
            
        except Exception as e:
            print(f"✗ Prediction failed: {e}")
            traceback.print_exc()
    else:
        print(f"✗ Model not found: {model_path}")

if __name__ == "__main__":
    print("Deepfake Detection Model Test")
    print("=" * 50)
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Python version: {sys.version}")
    
    # Run tests
    test_model_loading()
    test_input_generation()
    test_model_prediction()
    
    print("\n" + "=" * 50)
    print("Test completed!") 