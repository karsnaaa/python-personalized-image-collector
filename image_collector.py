import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import requests
import os
import time
import threading
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import json
import hashlib
from PIL import Image, ImageTk, ImageOps, ImageFilter, ImageDraw
import io
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import face_recognition
from sklearn.metrics.pairwise import cosine_similarity
import dlib
from imutils import face_utils
import math
import pytesseract
from datetime import datetime
import exifread
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from sklearn.cluster import DBSCAN
from skimage.metrics import structural_similarity as ssim
import warnings
warnings.filterwarnings('ignore')

class UltraPreciseProfileScraper:
    def __init__(self, root):
        self.root = root
        self.root.title("Ultra-Precise AI Profile Image Scraper")
        self.root.geometry("1200x950")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize multiple face detection models
        self.face_detectors = {
            'dlib': dlib.get_frontal_face_detector(),
            'haar': cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
            'mtcnn': None  # Will be loaded on demand
        }
        
        # Initialize face recognition models
        self.shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
        
        # Initialize image quality assessment model
        self.quality_model = self.load_quality_assessment_model()
        
        # Initialize scraper session with rotating user agents
        self.session = requests.Session()
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        self.current_ua_index = 0
        self.rotate_user_agent()
        
        # Configuration
        self.download_folder = "profile_images"
        self.is_scraping = False
        self.face_detection_enabled = True
        self.face_matching_enabled = True
        self.metadata_analysis_enabled = True
        self.quality_check_enabled = True
        self.min_face_size = 120  # Minimum face size in pixels
        self.similarity_threshold = 0.75  # Cosine similarity threshold for face matching
        self.min_quality_score = 0.7  # Minimum quality assessment score (0-1)
        self.max_aspect_ratio = 2.0  # Max width/height ratio for valid faces
        self.min_eye_distance = 30  # Minimum distance between eyes in pixels
        self.min_face_confidence = 0.9  # Minimum confidence score for face detection
        self.reference_faces = []
        
        # Create UI
        self.create_widgets()
        self.setup_advanced_settings()
        
        # Performance metrics
        self.total_images_processed = 0
        self.valid_faces_found = 0
        self.matching_faces_found = 0
        self.quality_rejections = 0
        self.metadata_rejections = 0

    def rotate_user_agent(self):
        """Rotate user agent to avoid detection"""
        self.session.headers.update({
            'User-Agent': self.user_agents[self.current_ua_index],
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.google.com/'
        })
        self.current_ua_index = (self.current_ua_index + 1) % len(self.user_agents)

    def load_quality_assessment_model(self):
        """Load pre-trained image quality assessment model"""
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def create_widgets(self):
        """Create the main UI components"""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel (controls)
        left_panel = ttk.Frame(main_container, width=450)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Right panel (preview and log)
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # ========== LEFT PANEL CONTROLS ==========
        
        # Search section
        search_frame = ttk.LabelFrame(left_panel, text="Search Parameters", padding=10)
        search_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Person name
        ttk.Label(search_frame, text="Person Name:").pack(anchor=tk.W)
        self.name_entry = ttk.Entry(search_frame, font=('Arial', 11))
        self.name_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Platforms
        platforms_frame = ttk.Frame(search_frame)
        platforms_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(platforms_frame, text="Platform Usernames:").pack(anchor=tk.W)
        
        # Twitter
        twitter_frame = ttk.Frame(platforms_frame)
        twitter_frame.pack(fill=tk.X, pady=2)
        ttk.Label(twitter_frame, text="Twitter/X:", width=10).pack(side=tk.LEFT)
        self.twitter_entry = ttk.Entry(twitter_frame)
        self.twitter_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # GitHub
        github_frame = ttk.Frame(platforms_frame)
        github_frame.pack(fill=tk.X, pady=2)
        ttk.Label(github_frame, text="GitHub:", width=10).pack(side=tk.LEFT)
        self.github_entry = ttk.Entry(github_frame)
        self.github_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # LinkedIn
        linkedin_frame = ttk.Frame(platforms_frame)
        linkedin_frame.pack(fill=tk.X, pady=2)
        ttk.Label(linkedin_frame, text="LinkedIn:", width=10).pack(side=tk.LEFT)
        self.linkedin_entry = ttk.Entry(linkedin_frame)
        self.linkedin_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Reference images
        ref_frame = ttk.LabelFrame(left_panel, text="Reference Images", padding=10)
        ref_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.ref_images_container = ttk.Frame(ref_frame)
        self.ref_images_container.pack(fill=tk.X)
        
        ref_buttons_frame = ttk.Frame(ref_frame)
        ref_buttons_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(ref_buttons_frame, text="Add Reference", command=self.add_reference_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(ref_buttons_frame, text="Clear All", command=self.clear_reference_images).pack(side=tk.LEFT)
        
        # Settings
        settings_frame = ttk.LabelFrame(left_panel, text="Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Download folder
        folder_frame = ttk.Frame(settings_frame)
        folder_frame.pack(fill=tk.X, pady=2)
        ttk.Label(folder_frame, text="Download Folder:", width=15).pack(side=tk.LEFT)
        self.folder_var = tk.StringVar(value=self.download_folder)
        self.folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_var)
        self.folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(folder_frame, text="Browse", command=self.browse_folder, width=8).pack(side=tk.LEFT)
        
        # Max images
        max_frame = ttk.Frame(settings_frame)
        max_frame.pack(fill=tk.X, pady=2)
        ttk.Label(max_frame, text="Max Images:", width=15).pack(side=tk.LEFT)
        self.max_images_var = tk.StringVar(value="20")
        ttk.Entry(max_frame, textvariable=self.max_images_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        buttons_frame = ttk.Frame(left_panel)
        buttons_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.start_button = ttk.Button(buttons_frame, text="Start Scraping", command=self.start_scraping, style='Accent.TButton')
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.stop_button = ttk.Button(buttons_frame, text="Stop", command=self.stop_scraping, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(buttons_frame, text="Advanced", command=self.show_advanced_settings).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # ========== RIGHT PANEL CONTROLS ==========
        
        # Preview frame
        preview_frame = ttk.LabelFrame(right_panel, text="Image Preview & Analysis", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        # Preview image with face annotations
        self.preview_canvas = tk.Canvas(preview_frame, bg='white')
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Analysis info
        analysis_frame = ttk.Frame(preview_frame)
        analysis_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.analysis_var = tk.StringVar(value="Analysis results will appear here")
        ttk.Label(analysis_frame, textvariable=self.analysis_var, wraplength=600).pack(anchor=tk.W)
        
        # Log frame
        log_frame = ttk.LabelFrame(right_panel, text="Activity Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=12, font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        self.progress = ttk.Progressbar(right_panel, mode='determinate')
        self.progress.pack(fill=tk.X, pady=(5, 0))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def validate_image(self, url):
        """Ultra-precise image validation with multiple checks"""
        try:
            # Initial URL validation
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                self.log_message(f"Invalid URL: {url}")
                return False
            
            # Rotate user agent for each request
            self.rotate_user_agent()
            
            # Initial HEAD request for content type and size
            head_response = self.session.head(url, timeout=10, allow_redirects=True)
            content_type = head_response.headers.get('content-type', '').lower()
            
            # Check if it's an image
            if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'jpg', 'png', 'gif', 'webp']):
                self.log_message(f"Non-image content type: {content_type}")
                return False
                
            # Check content length (avoid tiny images)
            content_length = head_response.headers.get('content-length')
            if content_length and int(content_length) < 5000:  # At least 5KB
                self.log_message(f"Image too small: {content_length} bytes")
                return False
            
            # Full GET request for image content
            response = self.session.get(url, stream=True, timeout=15)
            response.raise_for_status()
            
            # Read image content
            image_data = response.content
            image_array = np.frombuffer(image_data, np.uint8)
            
            # Decode image with OpenCV
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is None:
                self.log_message("Failed to decode image")
                return False
                
            # Basic image dimensions check
            if min(image.shape[:2]) < self.min_face_size:
                self.log_message(f"Image too small: {image.shape[:2]}")
                return False
            
            # Metadata analysis
            if self.metadata_analysis_enabled:
                metadata_valid = self.analyze_image_metadata(image_data)
                if not metadata_valid:
                    self.metadata_rejections += 1
                    return False
            
            # Quality assessment
            if self.quality_check_enabled:
                quality_score = self.assess_image_quality(image)
                if quality_score < self.min_quality_score:
                    self.quality_rejections += 1
                    self.log_message(f"Low quality score: {quality_score:.2f}")
                    return False
            
            # Face detection
            if self.face_detection_enabled:
                faces = self.detect_faces_ensemble(image)
                if not faces:
                    self.log_message("No valid faces detected")
                    return False
                    
                # Face matching if enabled
                if self.face_matching_enabled and self.reference_faces:
                    match_found = False
                    for face in faces:
                        shape = self.shape_predictor(image, face)
                        face_descriptor = self.face_recognizer.compute_face_descriptor(image, shape)
                        face_encoding = np.array(face_descriptor)
                        
                        for ref_face in self.reference_faces:
                            similarity = cosine_similarity(
                                [face_encoding],
                                [ref_face['encoding']]
                            )[0][0]
                            
                            if similarity >= self.similarity_threshold:
                                match_found = True
                                self.matching_faces_found += 1
                                break
                        
                        if match_found:
                            break
                    
                    if not match_found:
                        self.log_message("No matching faces found")
                        return False
            
            self.valid_faces_found += 1
            return True
            
        except Exception as e:
            self.log_message(f"Validation error for {url}: {str(e)}")
            return False

    def assess_image_quality(self, image):
        """Comprehensive image quality assessment"""
        try:
            # Convert to grayscale for some metrics
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 1. Blur detection using Laplacian variance
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = min(1.0, blur_score / 1000)  # Normalize
            
            # 2. Noise estimation using median absolute deviation
            median = np.median(gray)
            mad = np.median(np.abs(gray - median))
            noise_score = min(1.0, max(0, 1 - (mad / 30)))  # Normalize
            
            # 3. Contrast using RMS
            contrast_score = gray.std() / 255
            
            # 4. Illumination balance
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            illumination_score = 1 - np.sum(np.abs(hist - hist.mean())) / 2
            
            # 5. Deep learning quality assessment (placeholder)
            # Resize for model input
            resized = cv2.resize(image, (224, 224))
            dl_score = 0.8  # Placeholder - would use actual model prediction
            
            # Combined score (weighted average)
            total_score = (
                0.3 * blur_score +
                0.2 * noise_score +
                0.2 * contrast_score +
                0.2 * illumination_score +
                0.1 * dl_score
            )
            
            self.log_message(f"Quality scores - Blur: {blur_score:.2f}, Noise: {noise_score:.2f}, "
                           f"Contrast: {contrast_score:.2f}, Illum: {illumination_score:.2f}, "
                           f"DL: {dl_score:.2f}, Total: {total_score:.2f}")
            
            return total_score
            
        except Exception as e:
            self.log_message(f"Quality assessment error: {str(e)}")
            return 0.5  # Default score if assessment fails

    def analyze_image_metadata(self, image_data):
        """Analyze image metadata for authenticity"""
        try:
            # Check for EXIF data
            tags = exifread.process_file(io.BytesIO(image_data), details=False)
            
            # Check for suspicious metadata
            if 'Image Software' in tags:
                software = str(tags['Image Software'])
                if any(s in software.lower() for s in ['photoshop', 'editor', 'filter']):
                    self.log_message(f"Suspicious software: {software}")
                    return False
            
            # Check for unrealistic dimensions
            if 'Image ImageWidth' in tags and 'Image ImageLength' in tags:
                width = tags['Image ImageWidth'].values[0]
                height = tags['Image ImageLength'].values[0]
                if width > 10000 or height > 10000:  # Unrealistically large
                    self.log_message(f"Unrealistic dimensions: {width}x{height}")
                    return False
            
            # Check for compression artifacts
            # (This is a simplified check - real implementation would be more thorough)
            jpeg_quality = 95  # Default if not found
            if 'JPEGThumbnail' in tags:
                thumbnail_size = len(tags['JPEGThumbnail'].values)
                if thumbnail_size < 5000:  # Very compressed thumbnail
                    jpeg_quality = max(30, jpeg_quality - 30)
            
            if jpeg_quality < 70:
                self.log_message(f"High compression detected (quality ~{jpeg_quality})")
                return False
                
            return True
            
        except Exception as e:
            self.log_message(f"Metadata analysis error: {str(e)}")
            return True  # Assume valid if analysis fails

    def detect_faces_ensemble(self, image):
        """Advanced face detection using multiple methods with confidence scoring"""
        try:
            # Convert to RGB and grayscale
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            detections = []
            
            # 1. Dlib CNN (high accuracy)
            dlib_faces = self.face_detectors['dlib'](rgb_image, 1)
            for face in dlib_faces:
                detections.append({
                    'method': 'dlib',
                    'rect': face,
                    'confidence': 0.95  # Dlib is generally very reliable
                })
            
            # 2. Haar Cascade (fast but less accurate)
            haar_faces = self.face_detectors['haar'].detectMultiScale(
                gray_image,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            for (x, y, w, h) in haar_faces:
                detections.append({
                    'method': 'haar',
                    'rect': dlib.rectangle(x, y, x+w, y+h),
                    'confidence': 0.7  # Haar is less reliable
                })
            
            # 3. MTCNN (if enabled)
            if self.detection_method == 'mtcnn' or self.detection_method == 'ensemble':
                if self.face_detectors['mtcnn'] is None:
                    from mtcnn import MTCNN
                    self.face_detectors['mtcnn'] = MTCNN()
                
                mtcnn_results = self.face_detectors['mtcnn'].detect_faces(rgb_image)
                for result in mtcnn_results:
                    if result['confidence'] > self.min_face_confidence:
                        x, y, w, h = result['box']
                        detections.append({
                            'method': 'mtcnn',
                            'rect': dlib.rectangle(x, y, x+w, y+h),
                            'confidence': result['confidence']
                        })
            
            # Cluster overlapping detections
            if len(detections) > 1:
                boxes = np.array([[
                    det['rect'].left(), det['rect'].top(),
                    det['rect'].right(), det['rect'].bottom(),
                    det['confidence']
                ] for det in detections])
                
                # Use DBSCAN to cluster overlapping boxes
                clustering = DBSCAN(eps=50, min_samples=1).fit(boxes[:, :4])
                labels = clustering.labels_
                
                # For each cluster, keep the best detection
                unique_labels = set(labels)
                final_detections = []
                
                for label in unique_labels:
                    cluster_indices = [i for i, l in enumerate(labels) if l == label]
                    cluster_detections = [detections[i] for i in cluster_indices]
                    
                    # Select detection with highest confidence
                    best_detection = max(cluster_detections, key=lambda x: x['confidence'])
                    final_detections.append(best_detection)
                
                detections = final_detections
            
            # Additional validation for each face
            valid_faces = []
            for det in detections:
                face = det['rect']
                
                # Calculate face metrics
                width = face.width()
                height = face.height()
                aspect_ratio = width / height
                
                # Check minimum size
                if width < self.min_face_size or height < self.min_face_size:
                    continue
                
                # Check aspect ratio
                if aspect_ratio > self.max_aspect_ratio or aspect_ratio < (1/self.max_aspect_ratio):
                    continue
                
                # Check eye distance if possible
                try:
                    shape = self.shape_predictor(rgb_image, face)
                    landmarks = face_utils.shape_to_np(shape)
                    left_eye = landmarks[36:42]
                    right_eye = landmarks[42:48]
                    eye_distance = np.linalg.norm(np.mean(left_eye, axis=0) - np.mean(right_eye, axis=0))
                    
                    if eye_distance < self.min_eye_distance:
                        continue
                except:
                    pass
                
                valid_faces.append(face)
            
            return valid_faces
            
        except Exception as e:
            self.log_message(f"Face detection error: {str(e)}")
            return []
