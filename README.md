# Deepfake Detection System

A Flask-based web application for detecting deepfakes in images, videos, and audio files using pre-trained deep learning models.

## Issues Fixed

### 1. Model Loading Issues
- **Problem**: Models were failing to load due to compatibility issues
- **Solution**: Added `compile=False` parameter and better error handling
- **Result**: Models now load successfully with proper input shape detection

### 2. Input Shape Mismatch
- **Problem**: Audio preprocessing was creating wrong input shapes (e.g., `(1, 500, 56)` instead of `(1, 128, 109)`)
- **Solution**: Dynamic input shape detection and proper reshaping
- **Result**: Input data now matches model expectations

### 3. Audio Preprocessing Errors
- **Problem**: Complex audio feature extraction was causing shape mismatches
- **Solution**: Simplified MFCC extraction with proper padding/truncation
- **Result**: Audio files are now processed correctly

### 4. Model Path Issues
- **Problem**: Incorrect model file paths
- **Solution**: Updated to use `.keras` format models and correct paths
- **Result**: All models load from correct locations

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Model Files**:
   Ensure the following model files exist:
   - `deepfake-detection-tensorflow2-default-v1/image/XceptionModel.keras`
   - `deepfake-detection-tensorflow2-default-v1/audio/audio_classifier.h5`

## Usage

### Option 1: Run the Fixed Version (Recommended)
```bash
python app_fixed.py
```

### Option 2: Run the Original Version (Updated)
```bash
python app.py
```

### Option 3: Test Models Only
```bash
python test_models.py
```

## Features

### Supported File Types
- **Images**: PNG, JPG, JPEG, GIF, BMP, TIFF
- **Videos**: MP4, AVI, MOV, MKV, WEBM, FLV
- **Audio**: WAV, MP3, AAC, FLAC, OGG, M4A

### Model Architecture
- **Image/Video**: Xception-based CNN
- **Audio**: MFCC-based classifier
- **Input Sizes**: Dynamically detected from model files

### Detection Results
- **Real/Fake Classification**: Binary classification
- **Confidence Score**: Probability-based confidence
- **Processing Time**: Optimized for real-time use

## API Endpoints

### Health Check
```
GET /health
```
Returns model loading status and supported formats.

### File Upload
```
POST /upload
```
Upload a file for deepfake detection.

## Technical Details

### Model Loading Process
1. **Path Validation**: Check if model files exist
2. **Load with compile=False**: Avoid compilation issues
3. **Input Shape Detection**: Extract expected input dimensions
4. **Error Handling**: Graceful fallback for missing models

### Preprocessing Pipeline
1. **Image**: Resize to model input size, normalize to [0,1]
2. **Video**: Extract frames, resize, normalize
3. **Audio**: Extract MFCC features, pad/truncate to target length

### Error Handling
- **Model Loading**: Detailed error messages for missing/invalid models
- **Input Processing**: Graceful handling of unsupported file types
- **Prediction**: Comprehensive error reporting for debugging

## Troubleshooting

### Common Issues

1. **"Model not loaded" Error**:
   - Check if model files exist in the correct paths
   - Verify file permissions
   - Ensure TensorFlow version compatibility

2. **"Input shape incompatible" Error**:
   - The fixed version should handle this automatically
   - Check model input shapes using `test_models.py`

3. **Audio Processing Errors**:
   - Ensure librosa is installed correctly
   - Check audio file format support

### Debug Mode
Run with debug enabled to see detailed error messages:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## Performance Notes

- **Memory Usage**: Models are loaded once at startup
- **Processing Speed**: Optimized for real-time detection
- **File Size**: Maximum 100MB per file
- **Concurrent Users**: Flask development server limitations apply

## Future Improvements

1. **Model Ensemble**: Combine multiple models for better accuracy
2. **Real-time Processing**: WebSocket support for live video
3. **Batch Processing**: Support for multiple file uploads
4. **Model Updates**: Automatic model version management

## License

This project uses pre-trained models for deepfake detection. Please ensure compliance with model licenses and usage terms. 