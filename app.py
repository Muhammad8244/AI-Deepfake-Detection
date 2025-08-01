from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import librosa
from PIL import Image
import tensorflow as tf
import torch
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
    def __init__(self, image_model_path=None, audio_model_path=None, video_model_path=None):
        """
        Initialize the deepfake detector with multiple models for different media types
        """
        self.image_model = None
        self.audio_model = None
        self.video_model = None
        self.face_detector = None
        
        # Set default paths to your downloaded models
        self.image_model_path = image_model_path or "deepfake-detection-tensorflow2-default-v1/image/XceptionModel.keras"
        self.audio_model_path = audio_model_path or "deepfake-detection-tensorflow2-default-v1/audio/audio_classifier.h5"
        self.video_model_path = video_model_path or "deepfake-detection-tensorflow2-default-v1/image/XceptionModel.keras"
        
        self.load_models()
        self.setup_face_detector()
    
    def load_models(self):
        """
        Load the different models for image, audio, and video detection
        """
        # Load Image Model
        try:
            if os.path.exists(self.image_model_path):
                # Try loading with custom_objects for compatibility
                custom_objects = {}
                try:
                    self.image_model = tf.keras.models.load_model(self.image_model_path, custom_objects=custom_objects)
                except:
                    # Try loading with compile=False
                    self.image_model = tf.keras.models.load_model(self.image_model_path, compile=False)
                
                print(f"Image model loaded successfully: {self.image_model_path}")
                print(f"Image model input shape: {self.image_model.input_shape}")
            else:
                print(f"Image model not found: {self.image_model_path}")
        except Exception as e:
            print(f"Error loading image model: {e}")
            traceback.print_exc()
            
        # Load Audio Model
        try:
            if os.path.exists(self.audio_model_path):
                # Try loading with custom_objects for compatibility
                custom_objects = {}
                try:
                    self.audio_model = tf.keras.models.load_model(self.audio_model_path, custom_objects=custom_objects)
                except:
                    # Try loading with compile=False
                    self.audio_model = tf.keras.models.load_model(self.audio_model_path, compile=False)
                
                print(f"Audio model loaded successfully: {self.audio_model_path}")
                print(f"Audio model input shape: {self.audio_model.input_shape}")
            else:
                print(f"Audio model not found: {self.audio_model_path}")
        except Exception as e:
            print(f"Error loading audio model: {e}")
            traceback.print_exc()
            
        # Load Video Model (using same as image model or separate)
        try:
            if os.path.exists(self.video_model_path):
                # Try loading with custom_objects for compatibility
                custom_objects = {}
                try:
                    self.video_model = tf.keras.models.load_model(self.video_model_path, custom_objects=custom_objects)
                except:
                    # Try loading with compile=False
                    self.video_model = tf.keras.models.load_model(self.video_model_path, compile=False)
                
                print(f"Video model loaded successfully: {self.video_model_path}")
                print(f"Video model input shape: {self.video_model.input_shape}")
            else:
                # Fallback to image model for video processing
                self.video_model = self.image_model
                print("Using image model for video processing")
        except Exception as e:
            print(f"Error loading video model: {e}")
            traceback.print_exc()
            self.video_model = self.image_model
    
    def setup_face_detector(self):
        """
        Setup face detector for video processing (common in deepfake detection)
        """
        try:
            # Using OpenCV's DNN face detector (more accurate than Haar cascades)
            self.face_net = cv2.dnn.readNetFromTensorflow(
                'models/opencv_face_detector_uint8.pb',
                'models/opencv_face_detector.pbtxt'
            )
            print("Face detector loaded successfully")
        except:
            # Fallback to Haar cascade if DNN model not available
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                print("Haar cascade face detector loaded")
            except:
                print("Warning: No face detector available")
    
    def detect_faces(self, image):
        """
        Detect faces in image using available face detector
        Returns cropped face or original image if no face detected
        """
        try:
            if hasattr(self, 'face_net'):
                # Use DNN face detector
                h, w = image.shape[:2]
                blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
                self.face_net.setInput(blob)
                detections = self.face_net.forward()
                
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x, y, x1, y1 = box.astype(int)
                        return image[y:y1, x:x1]
                        
            elif hasattr(self, 'face_cascade'):
                # Use Haar cascade
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    return image[y:y+h, x:x+w]
            
            # Return original image if no face detected
            return image
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return image
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for deepfake detection model
        Based on common Kaggle deepfake detection approaches
        """
        try:
            # Load and convert image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect and crop face (important for deepfake detection)
            face_image = self.detect_faces(image)
            
            # Get model input shape
            if self.image_model and hasattr(self.image_model, 'input_shape'):
                target_shape = self.image_model.input_shape[1:3]  # Height, Width
                if len(target_shape) == 2:
                    target_size = (target_shape[1], target_shape[0])  # Width, Height
                else:
                    target_size = (224, 224)  # Default
            else:
                target_size = (224, 224)  # Default size
            
            face_image = cv2.resize(face_image, target_size)
            
            # Normalize pixel values (0-1 range for most models)
            face_image = face_image.astype(np.float32) / 255.0
            
            # Add batch dimension
            face_image = np.expand_dims(face_image, axis=0)
            
            return face_image
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def preprocess_video(self, video_path, max_frames=16):
        """
        Extract and preprocess frames from video for deepfake detection
        Focuses on face detection and extraction for better accuracy
        """
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Get model input shape
            if self.video_model and hasattr(self.video_model, 'input_shape'):
                target_shape = self.video_model.input_shape[1:3]  # Height, Width
                if len(target_shape) == 2:
                    target_size = (target_shape[1], target_shape[0])  # Width, Height
                else:
                    target_size = (224, 224)  # Default
            else:
                target_size = (224, 224)  # Default size
            
            # Sample frames evenly across the video
            frame_interval = max(1, total_frames // max_frames)
            
            while cap.read()[0] and len(frames) < max_frames:
                ret, frame = cap.read()
                if ret and frame_count % frame_interval == 0:
                    # Convert color space
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect and crop face
                    face_frame = self.detect_faces(frame)
                    
                    # Resize to model input size
                    face_frame = cv2.resize(face_frame, target_size)
                    
                    # Normalize
                    face_frame = face_frame.astype(np.float32) / 255.0
                    
                    frames.append(face_frame)
                
                frame_count += 1
            
            cap.release()
            
            if frames:
                # For sequence models, return as is
                # For single-frame models, take mean or use ensemble
                frames_array = np.array(frames)
                return frames_array
            
            return None
            
        except Exception as e:
            print(f"Error preprocessing video: {e}")
            return None
    
    def preprocess_audio(self, audio_path, duration=10):
        """
        Preprocess audio for deepfake detection
        Enhanced for the audio classifier models
        """
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=22050, duration=duration)
            
            # Get model input shape
            if self.audio_model and hasattr(self.audio_model, 'input_shape'):
                input_shape = self.audio_model.input_shape
                print(f"Audio model input shape: {input_shape}")
                
                # Handle different input shapes
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
            
            return features
            
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            traceback.print_exc()
            return None
    
    def predict(self, file_path, file_type):
        """
        Make prediction using the deepfake detection model
        Handles different input types and model formats
        """
        try:
            # Select the correct model based on file_type
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
                
                # Make prediction
                prediction = model.predict(processed_data, verbose=0)
                
                # Handle different output formats
                if len(prediction.shape) > 1 and prediction.shape[1] > 1:
                    # Multi-class output [real_prob, fake_prob]
                    fake_probability = float(prediction[0][1])
                else:
                    # Single output (sigmoid)
                    fake_probability = float(prediction[0][0])
                    
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
                
            elif file_type == 'audio':
                processed_data = self.preprocess_audio(file_path)
                if processed_data is None:
                    return {"error": "Failed to preprocess audio"}
                
                prediction = model.predict(processed_data, verbose=0)
                fake_probability = float(prediction[0][1] if prediction.shape[1] > 1 else prediction[0][0])
                
            else:
                return {"error": "Unsupported file type"}
            
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
                "model_info": {
                    "type": "deepfake_detector",
                    "input_size": "224x224" if file_type == 'image' else f"16_frames_224x224",
                    "architecture": "Xception-based" if hasattr(model, 'predict') else "Custom"
                },
                "timestamp": datetime.now().isoformat()
            }
            return result
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}. Please check your model format and input data."}

# Initialize detector with your downloaded models
# Update these paths to match your folder structure
detector = DeepfakeDetector(
    image_model_path="deepfake-detection-tensorflow2-default-v1/image/XceptionModel.keras",
    audio_model_path="deepfake-detection-tensorflow2-default-v1/audio/audio_classifier.h5",
    video_model_path="deepfake-detection-tensorflow2-default-v1/image/XceptionModel.keras"
)

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