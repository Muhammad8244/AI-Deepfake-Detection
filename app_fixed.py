from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import librosa
from PIL import Image
import tensorflow as tf
import json
from datetime import datetime
import tempfile
import shutil
import traceback

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {
    'image': {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'},
    'video': {'mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'},
    'audio': {'wav', 'mp3', 'aac', 'flac', 'ogg', 'm4a'}
}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class DeepfakeDetector:
    def __init__(self):
        """
        Initialize the deepfake detector with multiple models for different media types
        """
        self.image_model = None
        self.audio_model = None
        self.video_model = None
        
        # Model paths
        self.image_model_path = "deepfake-detection-tensorflow2-default-v1/image/XceptionModel.keras"
        self.audio_model_path = "deepfake-detection-tensorflow2-default-v1/audio/audio_classifier.h5"
        
        self.load_models()
    
    def load_models(self):
        """
        Load the different models for image, audio, and video detection
        """
        print("Loading models...")
        
        # Load Image Model
        try:
            if os.path.exists(self.image_model_path):
                self.image_model = tf.keras.models.load_model(self.image_model_path, compile=False)
                print(f"✓ Image model loaded: {self.image_model_path}")
                print(f"  Input shape: {self.image_model.input_shape}")
            else:
                print(f"✗ Image model not found: {self.image_model_path}")
        except Exception as e:
            print(f"✗ Error loading image model: {e}")
            
        # Load Audio Model
        try:
            if os.path.exists(self.audio_model_path):
                self.audio_model = tf.keras.models.load_model(self.audio_model_path, compile=False)
                print(f"✓ Audio model loaded: {self.audio_model_path}")
                print(f"  Input shape: {self.audio_model.input_shape}")
            else:
                print(f"✗ Audio model not found: {self.audio_model_path}")
        except Exception as e:
            print(f"✗ Error loading audio model: {e}")
        
        # Use image model for video processing
        self.video_model = self.image_model
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for deepfake detection model
        """
        try:
            # Load and convert image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get model input shape
            if self.image_model and hasattr(self.image_model, 'input_shape'):
                target_shape = self.image_model.input_shape[1:3]  # Height, Width
                target_size = (target_shape[1], target_shape[0])  # Width, Height
            else:
                target_size = (224, 224)  # Default
            
            # Resize image
            image = cv2.resize(image, target_size)
            
            # Normalize pixel values (0-1 range)
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def preprocess_audio(self, audio_path):
        """
        Preprocess audio for deepfake detection
        """
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=22050, duration=10)
            
            # Get model input shape
            if self.audio_model and hasattr(self.audio_model, 'input_shape'):
                input_shape = self.audio_model.input_shape
                print(f"Audio model expects: {input_shape}")
                
                if len(input_shape) == 3:  # (batch, time, features)
                    target_frames = input_shape[1]
                    n_features = input_shape[2]
                elif len(input_shape) == 4:  # (batch, height, width, channels)
                    target_frames = input_shape[1]
                    n_features = input_shape[2]
                else:
                    target_frames = 128
                    n_features = 109
            else:
                target_frames = 128
                n_features = 109
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_features, n_fft=2048, hop_length=512)
            
            # Pad or truncate to target frames
            if mfccs.shape[1] > target_frames:
                mfccs = mfccs[:, :target_frames]
            else:
                # Pad with zeros
                pad_width = target_frames - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
            
            # Reshape for model input
            if len(input_shape) == 3:
                # (batch, time, features)
                features = mfccs.T  # Transpose to (time, features)
                features = np.expand_dims(features, axis=0)  # Add batch dimension
            elif len(input_shape) == 4:
                # (batch, height, width, channels)
                features = mfccs.reshape(1, target_frames, n_features, 1)
            else:
                # Default shape
                features = mfccs.T
                features = np.expand_dims(features, axis=0)
            
            print(f"Generated audio features shape: {features.shape}")
            return features
            
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            traceback.print_exc()
            return None
    
    def preprocess_video(self, video_path, max_frames=16):
        """
        Extract and preprocess frames from video
        """
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Get model input shape
            if self.video_model and hasattr(self.video_model, 'input_shape'):
                target_shape = self.video_model.input_shape[1:3]
                target_size = (target_shape[1], target_shape[0])
            else:
                target_size = (224, 224)
            
            # Sample frames evenly
            frame_interval = max(1, total_frames // max_frames)
            
            while cap.read()[0] and len(frames) < max_frames:
                ret, frame = cap.read()
                if ret and frame_count % frame_interval == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, target_size)
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
                frame_count += 1
            
            cap.release()
            
            if frames:
                return np.array(frames)
            return None
            
        except Exception as e:
            print(f"Error preprocessing video: {e}")
            return None
    
    def predict(self, file_path, file_type):
        """
        Make prediction using the deepfake detection model
        """
        try:
            # Select the correct model
            if file_type == 'image':
                model = self.image_model
            elif file_type == 'audio':
                model = self.audio_model
            elif file_type == 'video':
                model = self.video_model
            else:
                return {"error": "Unsupported file type"}

            if model is None:
                return {"error": "Model not loaded. Please check model file path and format."}
            
            # Preprocess based on file type
            if file_type == 'image':
                processed_data = self.preprocess_image(file_path)
                if processed_data is None:
                    return {"error": "Failed to preprocess image"}
                
                prediction = model.predict(processed_data, verbose=0)
                
            elif file_type == 'video':
                processed_data = self.preprocess_video(file_path)
                if processed_data is None:
                    return {"error": "Failed to preprocess video"}
                
                frame_predictions = []
                for frame in processed_data:
                    frame_input = np.expand_dims(frame, axis=0)
                    pred = model.predict(frame_input, verbose=0)
                    frame_fake_prob = float(pred[0][1] if pred.shape[1] > 1 else pred[0][0])
                    frame_predictions.append(frame_fake_prob)
                
                fake_probability = max(frame_predictions)
                real_probability = 1.0 - fake_probability
                prediction_label = "FAKE" if fake_probability > 0.5 else "REAL"
                confidence = max(fake_probability, real_probability)
                
                return {
                    "fake_probability": fake_probability,
                    "real_probability": real_probability,
                    "prediction": prediction_label,
                    "confidence": confidence,
                    "model_info": {"type": "deepfake_detector"},
                    "timestamp": datetime.now().isoformat()
                }
                
            elif file_type == 'audio':
                processed_data = self.preprocess_audio(file_path)
                if processed_data is None:
                    return {"error": "Failed to preprocess audio"}
                
                prediction = model.predict(processed_data, verbose=0)
                
            else:
                return {"error": "Unsupported file type"}
            
            # Handle prediction output
            if len(prediction.shape) > 1 and prediction.shape[1] > 1:
                # Multi-class output [real_prob, fake_prob]
                fake_probability = float(prediction[0][1])
            else:
                # Single output (sigmoid)
                fake_probability = float(prediction[0][0])
            
            # Ensure probability is in valid range
            fake_probability = max(0.0, min(1.0, fake_probability))
            real_probability = 1.0 - fake_probability
            
            # Determine prediction and confidence
            prediction_label = "FAKE" if fake_probability > 0.5 else "REAL"
            confidence = max(fake_probability, real_probability)
            
            result = {
                "fake_probability": fake_probability,
                "real_probability": real_probability,
                "prediction": prediction_label,
                "confidence": confidence,
                "model_info": {"type": "deepfake_detector"},
                "timestamp": datetime.now().isoformat()
            }
            return result
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}. Please check your model format and input data."}

# Initialize detector
detector = DeepfakeDetector()

def allowed_file(filename, file_type):
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[file_type])

def get_file_type(filename):
    ext = filename.rsplit('.', 1)[1].lower()
    for file_type, extensions in ALLOWED_EXTENSIONS.items():
        if ext in extensions:
            return file_type
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    # Determine file type
    file_type = get_file_type(file.filename)
    if not file_type:
        return jsonify({'error': 'Unsupported file type'})
    
    if file and allowed_file(file.filename, file_type):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = timestamp + filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(file_path)
            
            # Make prediction
            result = detector.predict(file_path, file_type)
            
            # Clean up uploaded file
            try:
                os.remove(file_path)
            except:
                pass
            
            if 'error' in result:
                return jsonify({'error': result['error']})
            
            return jsonify({
                'success': True,
                'filename': file.filename,
                'file_type': file_type,
                'result': result
            })
            
        except Exception as e:
            # Clean up on error
            try:
                os.remove(file_path)
            except:
                pass
            return jsonify({'error': f'Processing failed: {str(e)}'})
    
    return jsonify({'error': 'Invalid file'})

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'image_model_loaded': detector.image_model is not None,
        'audio_model_loaded': detector.audio_model is not None,
        'video_model_loaded': detector.video_model is not None,
        'supported_formats': ALLOWED_EXTENSIONS
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 