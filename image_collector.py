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
from PIL import Image, ImageTk, ImageOps, ImageFilter
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
        
        # Initialize scraper session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.google.com/'
        })
        
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
        self.reference_faces = []
        
        # Create UI
        self.create_widgets()
        self.setup_advanced_settings()
        
        # Performance metrics
        self.total_images_processed = 0
        self.valid_faces_found = 0
        self.matching_faces_found = 0
    
    def load_quality_assessment_model(self):
        """Load pre-trained image quality assessment model"""
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Load custom weights (would need to train this model separately)
        # model.load_weights('quality_model_weights.h5')
        
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
    
    def setup_advanced_settings(self):
        """Create advanced settings window"""
        self.advanced_window = None
    
    def show_advanced_settings(self):
        """Show advanced settings window"""
        if self.advanced_window and self.advanced_window.winfo_exists():
            self.advanced_window.lift()
            return
            
        self.advanced_window = tk.Toplevel(self.root)
        self.advanced_window.title("Advanced Settings")
        self.advanced_window.geometry("600x500")
        self.advanced_window.resizable(False, False)
        
        # Face detection settings
        face_settings = ttk.LabelFrame(self.advanced_window, text="Face Detection Settings", padding=10)
        face_settings.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        # Detection method
        method_frame = ttk.Frame(face_settings)
        method_frame.pack(fill=tk.X, pady=2)
        ttk.Label(method_frame, text="Detection Method:").pack(side=tk.LEFT)
        self.detection_method_var = tk.StringVar(value="ensemble")
        methods = [("Ensemble (recommended)", "ensemble"),
                  ("Dlib CNN", "dlib"),
                  ("Haar Cascade", "haar"),
                  ("MTCNN", "mtcnn")]
        for text, mode in methods:
            ttk.Radiobutton(method_frame, text=text, variable=self.detection_method_var, 
                           value=mode).pack(side=tk.LEFT, padx=5)
        
        # Min face size
        min_face_frame = ttk.Frame(face_settings)
        min_face_frame.pack(fill=tk.X, pady=2)
        ttk.Label(min_face_frame, text="Minimum Face Size (px):").pack(side=tk.LEFT)
        self.min_face_var = tk.IntVar(value=self.min_face_size)
        ttk.Scale(min_face_frame, from_=50, to=200, variable=self.min_face_var, 
                 command=lambda v: self.min_face_size_label.config(text=f"{int(float(v))}px")).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.min_face_size_label = ttk.Label(min_face_frame, text=f"{self.min_face_size}px", width=5)
        self.min_face_size_label.pack(side=tk.LEFT)
        
        # Min eye distance
        eye_dist_frame = ttk.Frame(face_settings)
        eye_dist_frame.pack(fill=tk.X, pady=2)
        ttk.Label(eye_dist_frame, text="Min Eye Distance (px):").pack(side=tk.LEFT)
        self.min_eye_var = tk.IntVar(value=self.min_eye_distance)
        ttk.Scale(eye_dist_frame, from_=10, to=100, variable=self.min_eye_var, 
                 command=lambda v: self.min_eye_label.config(text=f"{int(float(v))}px")).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.min_eye_label = ttk.Label(eye_dist_frame, text=f"{self.min_eye_distance}px", width=5)
        self.min_eye_label.pack(side=tk.LEFT)
        
        # Max aspect ratio
        aspect_frame = ttk.Frame(face_settings)
        aspect_frame.pack(fill=tk.X, pady=2)
        ttk.Label(aspect_frame, text="Max Aspect Ratio:").pack(side=tk.LEFT)
        self.aspect_var = tk.DoubleVar(value=self.max_aspect_ratio)
        ttk.Scale(aspect_frame, from_=1.0, to=3.0, variable=self.aspect_var, 
                 command=lambda v: self.aspect_label.config(text=f"{float(v):.1f}")).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.aspect_label = ttk.Label(aspect_frame, text=f"{self.max_aspect_ratio:.1f}", width=5)
        self.aspect_label.pack(side=tk.LEFT)
        
        # Face matching settings
        match_settings = ttk.LabelFrame(self.advanced_window, text="Face Matching Settings", padding=10)
        match_settings.pack(fill=tk.X, padx=10, pady=5)
        
        # Similarity threshold
        similarity_frame = ttk.Frame(match_settings)
        similarity_frame.pack(fill=tk.X, pady=2)
        ttk.Label(similarity_frame, text="Similarity Threshold:").pack(side=tk.LEFT)
        self.similarity_var = tk.DoubleVar(value=self.similarity_threshold)
        ttk.Scale(similarity_frame, from_=0.5, to=0.95, variable=self.similarity_var, 
                 command=lambda v: self.similarity_label.config(text=f"{float(v):.2f}")).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.similarity_label = ttk.Label(similarity_frame, text=f"{self.similarity_threshold:.2f}", width=5)
        self.similarity_label.pack(side=tk.LEFT)
        
        # Image quality settings
        quality_settings = ttk.LabelFrame(self.advanced_window, text="Image Quality Settings", padding=10)
        quality_settings.pack(fill=tk.X, padx=10, pady=5)
        
        # Min quality score
        quality_frame = ttk.Frame(quality_settings)
        quality_frame.pack(fill=tk.X, pady=2)
        ttk.Label(quality_frame, text="Min Quality Score:").pack(side=tk.LEFT)
        self.quality_var = tk.DoubleVar(value=self.min_quality_score)
        ttk.Scale(quality_frame, from_=0.1, to=0.9, variable=self.quality_var, 
                 command=lambda v: self.quality_label.config(text=f"{float(v):.2f}")).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.quality_label = ttk.Label(quality_frame, text=f"{self.min_quality_score:.2f}", width=5)
        self.quality_label.pack(side=tk.LEFT)
        
        # Save button
        save_frame = ttk.Frame(self.advanced_window)
        save_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(save_frame, text="Save Settings", command=self.save_advanced_settings).pack(side=tk.RIGHT)
    
    def save_advanced_settings(self):
        """Save advanced settings"""
        self.min_face_size = self.min_face_var.get()
        self.similarity_threshold = self.similarity_var.get()
        self.min_quality_score = self.quality_var.get()
        self.min_eye_distance = self.min_eye_var.get()
        self.max_aspect_ratio = self.aspect_var.get()
        self.detection_method = self.detection_method_var.get()
        
        self.log_message(f"Advanced settings saved:")
        self.log_message(f"- Min face size: {self.min_face_size}px")
        self.log_message(f"- Min eye distance: {self.min_eye_distance}px")
        self.log_message(f"- Max aspect ratio: {self.max_aspect_ratio:.1f}")
        self.log_message(f"- Similarity threshold: {self.similarity_threshold:.2f}")
        self.log_message(f"- Min quality score: {self.min_quality_score:.2f}")
        self.log_message(f"- Detection method: {self.detection_method}")
        
        self.advanced_window.destroy()
        self.advanced_window = None
    
    def add_reference_image(self):
        """Add reference image with enhanced processing"""
        file_path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp")]
        )
        
        if file_path:
            try:
                # Load image
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError("Could not read image file")
                
                # Convert to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect faces using ensemble method
                faces = self.detect_faces_ensemble(rgb_image)
                if len(faces) == 0:
                    raise ValueError("No faces found in the image")
                
                # Get the best quality face
                best_face = self.select_best_face(rgb_image, faces)
                
                # Predict landmarks
                shape = self.shape_predictor(rgb_image, best_face)
                
                # Compute face descriptor
                face_descriptor = self.face_recognizer.compute_face_descriptor(rgb_image, shape)
                face_encoding = np.array(face_descriptor)
                
                # Calculate quality metrics
                landmarks = face_utils.shape_to_np(shape)
                left_eye = landmarks[36:42]
                right_eye = landmarks[42:48]
                eye_distance = np.linalg.norm(np.mean(left_eye, axis=0) - np.mean(right_eye, axis=0))
                
                # Store the reference face
                self.reference_faces.append({
                    'encoding': face_encoding,
                    'image': image,
                    'path': file_path,
                    'landmarks': shape,
                    'eye_distance': eye_distance,
                    'face_rect': best_face
                })
                
                # Display thumbnail with annotations
                self.display_reference_thumbnail(file_path, len(self.reference_faces), best_face, shape)
                
                self.log_message(f"Added reference image: {os.path.basename(file_path)}")
                self.log_message(f"Face detected with eye distance: {eye_distance:.1f}px")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process reference image: {str(e)}")
    
    def detect_faces_ensemble(self, image):
        """Detect faces using multiple methods for better accuracy"""
        # Convert to grayscale for Haar cascade
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect with each method
        detections = []
        
        # 1. Dlib CNN (most accurate but slower)
        dlib_faces = self.face_detectors['dlib'](image, 1)
        for face in dlib_faces:
            detections.append({
                'method': 'dlib',
                'rect': face,
                'confidence': 1.0  # Dlib doesn't provide confidence scores
            })
        
        # 2. Haar Cascade (faster but less accurate)
        haar_faces = self.face_detectors['haar'].detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in haar_faces:
            detections.append({
                'method': 'haar',
                'rect': dlib.rectangle(x, y, x+w, y+h),
                'confidence': 1.0  # Haar doesn't provide confidence
            })
        
        # 3. MTCNN (on-demand loading)
        if self.detection_method == 'mtcnn' or self.detection_method == 'ensemble':
            if self.face_detectors['mtcnn'] is None:
                from mtcnn import MTCNN
                self.face_detectors['mtcnn'] = MTCNN()
            
            mtcnn_results = self.face_detectors['mtcnn'].detect_faces(image)
            for result in mtcnn_results:
                if result['confidence'] > 0.9:  # Only high-confidence detections
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
                det['rect'].right(), det['rect'].bottom()
            ] for det in detections])
            
            # Use DBSCAN to cluster overlapping boxes
            clustering = DBSCAN(eps=50, min_samples=1).fit(boxes)
            labels = clustering.labels_
            
            # For each cluster, keep the best detection
            unique_labels = set(labels)
            final_detections = []
            
            for label in unique_labels:
                cluster_detections = [detections[i] for i in range(len(detections)) if labels[i] == label]
                
                # Prefer MTCNN if available, then Dlib, then Haar
                method_priority = {'mtcnn': 3, 'dlib': 2, 'haar': 1}
                best_detection = max(cluster_detections, key=lambda x: (
                    method_priority.get(x['method'], 0),
                    x.get('confidence', 0)
                ))
                
                final_detections.append(best_detection)
            
            detections = final_detections
        
        return [det['rect'] for det in detections]
    
    def select_best_face(self, image, faces):
        """Select the best face from multiple detections"""
        if len(faces) == 1:
            return faces[0]
        
        # Score each face based on size, position, and aspect ratio
        img_height, img_width = image.shape[:2]
        center_x, center_y = img_width // 2, img_height // 2
        
        best_score = -1
        best_face = None
        
        for face in faces:
            # Calculate face metrics
            width = face.width()
            height = face.height()
            aspect_ratio = width / height
            area = width * height
            
            # Calculate distance from image center
            face_center_x = (face.left() + face.right()) // 2
            face_center_y = (face.top() + face.bottom()) // 2
            dist_from_center = math.sqrt(
                (face_center_x - center_x)**2 + 
                (face_center_y - center_y)**2
            )
            
            # Normalize distance (0-1 where 0 is center)
            max_dist = math.sqrt(center_x**2 + center_y**2)
            norm_dist = dist_from_center / max_dist if max_dist > 0 else 0
            
            # Calculate score (higher is better)
            size_score = min(1.0, area / (self.min_face_size ** 2))  # Reward larger faces
            aspect_score = 1.0 - min(1.0, abs(aspect_ratio - 1.0))  # Reward square-ish faces
            center_score = 1.0 - norm_dist  # Reward centered faces
            
            total_score = 0.5 * size_score + 0.3 * aspect_score + 0.2 * center_score
            
            if total_score > best_score:
                best_score = total_score
                best_face = face
        
        return best_face
    
    def display_reference_thumbnail(self, file_path, index, face_rect, shape):
        """Display annotated reference thumbnail"""
        try:
            # Create frame if needed
            if not hasattr(self, 'ref_thumbnail_frames'):
                self.ref_thumbnail_frames = []
                self.ref_thumbnail_canvases = []
                self.ref_thumbnail_texts = []
            
            # Create new frame
            frame = ttk.Frame(self.ref_images_container)
            frame.pack(fill=tk.X, pady=2)
            self.ref_thumbnail_frames.append(frame)
            
            # Create canvas for annotated image
            canvas = tk.Canvas(frame, width=100, height=100, bg='white')
            canvas.pack(side=tk.LEFT, padx=5)
            self.ref_thumbnail_canvases.append(canvas)
            
            # Load image
            img = Image.open(file_path)
            img.thumbnail((100, 100))
            
            # Draw face rectangle and landmarks
            draw = ImageDraw.Draw(img)
            
            # Scale face rect to thumbnail size
            scale_x = 100 / img.width
            scale_y = 100 / img.height
            
            # Draw face rectangle
            draw.rectangle([
                (face_rect.left() * scale_x, face_rect.top() * scale_y),
                (face_rect.right() * scale_x, face_rect.bottom() * scale_y)
            ], outline="red", width=1)
            
            # Draw landmarks
            landmarks = face_utils.shape_to_np(shape)
            for (x, y) in landmarks:
                draw.ellipse([(x*scale_x-1, y*scale_y-1), (x*scale_x+1, y*scale_y+1)], 
                            fill="blue")
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Display on canvas
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.image = photo  # Keep reference
            
            # Add index text
            text = ttk.Label(frame, text=f"Ref #{index}", font=('Arial', 8))
            text.pack(side=tk.LEFT)
            self.ref_thumbnail_texts.append(text)
            
            # Add remove button
            remove_btn = ttk.Button(frame, text="×", width=2, 
                                   command=lambda i=index-1: self.remove_reference_image(i))
            remove_btn.pack(side=tk.RIGHT, padx=5)
            
        except Exception as e:
            self.log_message(f"Error displaying thumbnail: {str(e)}")
    
    # [Rest of the methods including remove_reference_image, clear_reference_images, 
    # toggle_face_detection, toggle_face_matching, log_message, clear_log, 
    # browse_folder, open_download_folder, start_scraping, stop_scraping would be here]
    
    def validate_image(self, url):
        """Ultra-precise image validation with multiple checks"""
        try:
            # [Implementation would include all the advanced validation steps]
            pass
    
    def assess_image_quality(self, image):
        """Assess image quality using multiple metrics"""
        # [Implementation would include blur detection, noise analysis, etc.]
        pass
    
    def analyze_image_metadata(self, image_data):
        """Analyze image metadata for authenticity"""
        # [Implementation would check EXIF data, compression artifacts, etc.]
        pass
    
    def scrape_images(self, person_name):
        """Main scraping method with enhanced validation"""
        # [Implementation would include all the scraping logic with new validation]
        pass
    
    def download_and_process_image(self, img_info, person_name, current, total):
        """Download and process with all validation steps"""
        # [Implementation would include all processing steps]
        pass
    
    def update_image_preview(self, filepath):
        """Update preview with detailed annotations"""
        # [Implementation would show face boxes, landmarks, quality info]
        pass

def main():
    root = tk.Tk()
    app = UltraPreciseProfileScraper(root)
    root.mainloop()

if __name__ == "__main__":
    main()
