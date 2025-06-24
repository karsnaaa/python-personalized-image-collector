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
from PIL import Image, ImageTk, ImageOps
import io
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import face_recognition
from sklearn.metrics.pairwise import cosine_similarity
import dlib
from imutils import face_utils
import math

class AdvancedProfileImageScraper:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced AI Profile Image Scraper")
        self.root.geometry("1100x900")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize face detection models
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # You'll need to download this
        self.face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")  # Download this too
        
        # Initialize scraper session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.google.com/'
        })
        
        self.download_folder = "profile_images"
        self.is_scraping = False
        self.face_detection_enabled = True
        self.face_matching_enabled = True
        self.reference_faces = []
        self.min_face_size = 100  # Minimum face size in pixels
        self.similarity_threshold = 0.7  # Cosine similarity threshold for face matching
        
        self.create_widgets()
        self.setup_advanced_settings()
    
    def create_widgets(self):
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel (controls)
        left_panel = ttk.Frame(main_container, width=400)
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
        
        # Face detection
        face_frame = ttk.Frame(settings_frame)
        face_frame.pack(fill=tk.X, pady=2)
        ttk.Label(face_frame, text="Face Detection:", width=15).pack(side=tk.LEFT)
        self.face_detect_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(face_frame, variable=self.face_detect_var, command=self.toggle_face_detection).pack(side=tk.LEFT)
        
        # Face matching
        match_frame = ttk.Frame(settings_frame)
        match_frame.pack(fill=tk.X, pady=2)
        ttk.Label(match_frame, text="Face Matching:", width=15).pack(side=tk.LEFT)
        self.face_match_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(match_frame, variable=self.face_match_var, command=self.toggle_face_matching).pack(side=tk.LEFT)
        
        # Action buttons
        buttons_frame = ttk.Frame(left_panel)
        buttons_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.start_button = ttk.Button(buttons_frame, text="Start Scraping", command=self.start_scraping, style='Accent.TButton')
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.stop_button = ttk.Button(buttons_frame, text="Stop", command=self.stop_scraping, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(buttons_frame, text="Open Folder", command=self.open_download_folder).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # ========== RIGHT PANEL CONTROLS ==========
        
        # Preview frame
        preview_frame = ttk.LabelFrame(right_panel, text="Image Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        self.preview_label = ttk.Label(preview_frame, text="Image preview will appear here", anchor=tk.CENTER)
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        
        # Log frame
        log_frame = ttk.LabelFrame(right_panel, text="Activity Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, font=('Consolas', 9))
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
        
        # Create but don't show yet
        if hasattr(self, 'advanced_settings_button'):
            return
            
        self.advanced_settings_button = ttk.Button(self.root, text="⚙ Advanced Settings", 
                                                 command=self.show_advanced_settings, style='Small.TButton')
        self.advanced_settings_button.place(relx=1.0, rely=0.0, anchor=tk.NE, x=-10, y=10)
    
    def show_advanced_settings(self):
        """Show advanced settings window"""
        if self.advanced_window and self.advanced_window.winfo_exists():
            self.advanced_window.lift()
            return
            
        self.advanced_window = tk.Toplevel(self.root)
        self.advanced_window.title("Advanced Settings")
        self.advanced_window.geometry("500x400")
        self.advanced_window.resizable(False, False)
        
        # Face detection settings
        face_settings = ttk.LabelFrame(self.advanced_window, text="Face Detection Settings", padding=10)
        face_settings.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        # Min face size
        min_face_frame = ttk.Frame(face_settings)
        min_face_frame.pack(fill=tk.X, pady=2)
        ttk.Label(min_face_frame, text="Minimum Face Size (px):").pack(side=tk.LEFT)
        self.min_face_var = tk.IntVar(value=self.min_face_size)
        ttk.Scale(min_face_frame, from_=50, to=200, variable=self.min_face_var, 
                 command=lambda v: self.min_face_size_label.config(text=f"{int(float(v))}px")).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.min_face_size_label = ttk.Label(min_face_frame, text=f"{self.min_face_size}px", width=5)
        self.min_face_size_label.pack(side=tk.LEFT)
        
        # Similarity threshold
        similarity_frame = ttk.Frame(face_settings)
        similarity_frame.pack(fill=tk.X, pady=2)
        ttk.Label(similarity_frame, text="Similarity Threshold:").pack(side=tk.LEFT)
        self.similarity_var = tk.DoubleVar(value=self.similarity_threshold)
        ttk.Scale(similarity_frame, from_=0.5, to=0.95, variable=self.similarity_var, 
                 command=lambda v: self.similarity_label.config(text=f"{float(v):.2f}")).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.similarity_label = ttk.Label(similarity_frame, text=f"{self.similarity_threshold:.2f}", width=5)
        self.similarity_label.pack(side=tk.LEFT)
        
        # Search settings
        search_settings = ttk.LabelFrame(self.advanced_window, text="Search Settings", padding=10)
        search_settings.pack(fill=tk.X, padx=10, pady=5)
        
        # Search depth
        depth_frame = ttk.Frame(search_settings)
        depth_frame.pack(fill=tk.X, pady=2)
        ttk.Label(depth_frame, text="Search Depth:").pack(side=tk.LEFT)
        self.search_depth_var = tk.IntVar(value=2)
        ttk.Combobox(depth_frame, textvariable=self.search_depth_var, values=[1, 2, 3], width=5).pack(side=tk.LEFT, padx=5)
        
        # Timeout
        timeout_frame = ttk.Frame(search_settings)
        timeout_frame.pack(fill=tk.X, pady=2)
        ttk.Label(timeout_frame, text="Request Timeout (s):").pack(side=tk.LEFT)
        self.timeout_var = tk.IntVar(value=15)
        ttk.Entry(timeout_frame, textvariable=self.timeout_var, width=5).pack(side=tk.LEFT, padx=5)
        
        # Save button
        save_frame = ttk.Frame(self.advanced_window)
        save_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(save_frame, text="Save Settings", command=self.save_advanced_settings).pack(side=tk.RIGHT)
    
    def save_advanced_settings(self):
        """Save advanced settings"""
        self.min_face_size = self.min_face_var.get()
        self.similarity_threshold = self.similarity_var.get()
        self.log_message(f"Advanced settings saved: Min Face Size={self.min_face_size}px, Similarity={self.similarity_threshold:.2f}")
        self.advanced_window.destroy()
        self.advanced_window = None
    
    def add_reference_image(self):
        """Add reference image for face matching"""
        file_path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp")]
        )
        
        if file_path:
            try:
                # Load and process the image
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError("Could not read image file")
                
                # Convert to RGB (dlib uses RGB)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                faces = self.face_detector(rgb_image, 1)
                if len(faces) == 0:
                    raise ValueError("No faces found in the image")
                
                # Get the largest face
                largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
                
                # Predict landmarks
                shape = self.shape_predictor(rgb_image, largest_face)
                
                # Compute face descriptor
                face_descriptor = self.face_recognizer.compute_face_descriptor(rgb_image, shape)
                
                # Convert to numpy array
                face_encoding = np.array(face_descriptor)
                
                # Store the encoding and original image
                self.reference_faces.append({
                    'encoding': face_encoding,
                    'image': image,
                    'path': file_path,
                    'landmarks': shape
                })
                
                # Display thumbnail
                self.display_reference_thumbnail(file_path, len(self.reference_faces))
                
                self.log_message(f"Added reference image: {os.path.basename(file_path)}")
                self.log_message(f"Current reference faces: {len(self.reference_faces)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process reference image: {str(e)}")
    
    def display_reference_thumbnail(self, file_path, index):
        """Display a thumbnail of the reference image"""
        try:
            # Create a thumbnail frame if it doesn't exist
            if not hasattr(self, 'ref_thumbnail_frames'):
                self.ref_thumbnail_frames = []
                self.ref_thumbnail_labels = []
                self.ref_thumbnail_texts = []
            
            # Create new frame for this thumbnail
            frame = ttk.Frame(self.ref_images_container)
            frame.pack(fill=tk.X, pady=2)
            self.ref_thumbnail_frames.append(frame)
            
            # Load and resize image
            img = Image.open(file_path)
            img.thumbnail((80, 80))
            photo = ImageTk.PhotoImage(img)
            
            # Create label for image
            label = ttk.Label(frame, image=photo)
            label.image = photo  # Keep reference
            label.pack(side=tk.LEFT, padx=5)
            self.ref_thumbnail_labels.append(label)
            
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
    
    def remove_reference_image(self, index):
        """Remove a reference image by index"""
        if 0 <= index < len(self.reference_faces):
            # Remove from data
            removed = self.reference_faces.pop(index)
            
            # Remove from UI
            self.ref_thumbnail_frames[index].destroy()
            self.ref_thumbnail_frames.pop(index)
            self.ref_thumbnail_labels.pop(index)
            self.ref_thumbnail_texts.pop(index)
            
            # Update remaining thumbnails' text
            for i, text in enumerate(self.ref_thumbnail_texts):
                text.config(text=f"Ref #{i+1}")
            
            self.log_message(f"Removed reference image: {os.path.basename(removed['path'])}")
    
    def clear_reference_images(self):
        """Clear all reference images"""
        self.reference_faces = []
        
        if hasattr(self, 'ref_thumbnail_frames'):
            for frame in self.ref_thumbnail_frames:
                frame.destroy()
            
            self.ref_thumbnail_frames = []
            self.ref_thumbnail_labels = []
            self.ref_thumbnail_texts = []
        
        self.log_message("Cleared all reference images")
    
    def toggle_face_detection(self):
        """Toggle face detection on/off"""
        self.face_detection_enabled = self.face_detect_var.get()
        status = "enabled" if self.face_detection_enabled else "disabled"
        self.log_message(f"Face detection {status}")
    
    def toggle_face_matching(self):
        """Toggle face matching on/off"""
        self.face_matching_enabled = self.face_match_var.get()
        status = "enabled" if self.face_matching_enabled else "disabled"
        self.log_message(f"Face matching {status}")
    
    def log_message(self, message):
        """Add message to log area"""
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_log(self):
        """Clear the log area"""
        self.log_text.delete(1.0, tk.END)
    
    def browse_folder(self):
        """Browse for download folder"""
        folder = filedialog.askdirectory(initialdir=self.folder_var.get())
        if folder:
            self.folder_var.set(folder)
            self.download_folder = folder
    
    def open_download_folder(self):
        """Open the download folder in file explorer"""
        if os.path.exists(self.download_folder):
            if os.name == 'nt':  # Windows
                os.startfile(self.download_folder)
            elif os.name == 'posix':  # macOS and Linux
                os.system(f'open "{self.download_folder}"')
        else:
            messagebox.showwarning("Folder Not Found", f"Folder '{self.download_folder}' does not exist.")
    
    def start_scraping(self):
        """Start the scraping process"""
        person_name = self.name_entry.get().strip()
        if not person_name:
            messagebox.showerror("Error", "Please enter a person name.")
            return
        
        self.is_scraping = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress.start()
        self.status_var.set("Scraping in progress...")
        
        # Start scraping in a separate thread
        self.scraping_thread = threading.Thread(target=self.scrape_images, args=(person_name,))
        self.scraping_thread.daemon = True
        self.scraping_thread.start()
    
    def stop_scraping(self):
        """Stop the scraping process"""
        self.is_scraping = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress.stop()
        self.status_var.set("Scraping stopped")
        self.log_message("Scraping stopped by user")
    
    def validate_image(self, url: str) -> bool:
        """Advanced image validation with multiple checks"""
        try:
            # Quick HEAD request to check if URL exists and is an image
            response = self.session.head(url, timeout=self.timeout_var.get())
            content_type = response.headers.get('content-type', '').lower()
            
            # Check if it's an image
            if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'jpg', 'png', 'gif', 'webp']):
                return False
                
            # Check content length (avoid tiny images)
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) < 2000:  # At least 2KB
                return False
            
            # If face detection is disabled, skip further checks
            if not self.face_detection_enabled:
                return True
                
            # Download the full image for face detection
            response = self.session.get(url, timeout=self.timeout_var.get())
            response.raise_for_status()
            
            image_data = response.content
            
            # Convert to numpy array
            image = np.array(Image.open(io.BytesIO(image_data)))
            
            # Convert to RGB (dlib uses RGB)
            if len(image.shape) == 2:  # Grayscale
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = self.face_detector(rgb_image, 1)
            
            # Check if we found any faces
            if len(faces) == 0:
                return False
                
            # Check face sizes
            img_height, img_width = rgb_image.shape[:2]
            min_dim = min(img_height, img_width)
            
            for face in faces:
                # Calculate face area
                face_area = face.width() * face.height()
                
                # Calculate minimum required area based on settings
                min_area = self.min_face_size ** 2
                
                if face_area < min_area:
                    continue  # Skip small faces
                
                # If we have reference faces, check for matches
                if self.face_matching_enabled and self.reference_faces:
                    # Predict landmarks
                    shape = self.shape_predictor(rgb_image, face)
                    
                    # Compute face descriptor
                    face_descriptor = self.face_recognizer.compute_face_descriptor(rgb_image, shape)
                    face_encoding = np.array(face_descriptor)
                    
                    # Compare with reference faces
                    for ref_face in self.reference_faces:
                        # Calculate cosine similarity
                        similarity = cosine_similarity(
                            [face_encoding],
                            [ref_face['encoding']]
                        )[0][0]
                        
                        if similarity >= self.similarity_threshold:
                            return True
                    
                    # If we get here, no matches found
                    return False
                else:
                    # No reference faces or matching disabled, any valid face is OK
                    return True
                    
            # If we get here, all faces were too small
            return False
            
        except Exception as e:
            self.log_message(f"Validation error for {url}: {str(e)}")
            return False
    
    def scrape_images(self, person_name):
        """Main scraping method with enhanced search capabilities"""
        try:
            self.download_folder = self.folder_var.get()
            os.makedirs(self.download_folder, exist_ok=True)
            
            # Collect platform info
            platforms = {}
            if self.twitter_entry.get().strip():
                platforms['twitter'] = self.twitter_entry.get().strip()
            if self.github_entry.get().strip():
                platforms['github'] = self.github_entry.get().strip()
            if self.linkedin_entry.get().strip():
                platforms['linkedin'] = self.linkedin_entry.get().strip()
            
            max_images = int(self.max_images_var.get() or 20)
            
            self.log_message(f"Starting advanced scrape for: {person_name}")
            self.log_message(f"Max images: {max_images}")
            if self.face_detection_enabled:
                self.log_message(f"Face detection: ON (min size: {self.min_face_size}px)")
                if self.face_matching_enabled and self.reference_faces:
                    self.log_message(f"Face matching: ON (threshold: {self.similarity_threshold:.2f})")
                else:
                    self.log_message("Face matching: OFF (no reference faces)")
            else:
                self.log_message("Face detection: OFF")
            
            all_images = []
            
            # Check specific platforms first
            platform_results = self.scrape_platforms(platforms, person_name)
            all_images.extend(platform_results)
            
            # Search Google Images if needed
            if self.is_scraping and len(all_images) < max_images:
                google_images = self.search_google_images(person_name, max_images - len(all_images))
                all_images.extend(google_images)
            
            # Search alternative sources if needed
            if self.is_scraping and len(all_images) < max_images:
                alt_images = self.search_alternative_sources(person_name, max_images - len(all_images))
                all_images.extend(alt_images)
            
            # Validate and filter images
            valid_images = []
            for img in all_images:
                if not self.is_scraping:
                    break
                
                if self.validate_image(img['url']):
                    valid_images.append(img)
                    self.log_message(f"✓ Valid image from {img['source']}")
                else:
                    reason = "no valid face" if self.face_detection_enabled else "invalid/inaccessible"
                    self.log_message(f"✗ Rejected image from {img['source']} ({reason})")
            
            # Download images with threading
            downloaded_count = 0
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for i, img_info in enumerate(valid_images[:max_images]):
                    if not self.is_scraping:
                        break
                    
                    futures.append(executor.submit(self.download_and_process_image, img_info, person_name, i+1, len(valid_images)))
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        downloaded_count += 1
                        self.progress['value'] = (downloaded_count / min(len(valid_images), max_images)) * 100
            
            if self.is_scraping:
                self.log_message(f"Scraping completed! Downloaded {downloaded_count} images.")
                self.status_var.set(f"Completed - {downloaded_count} images downloaded")
                messagebox.showinfo("Complete", f"Successfully downloaded {downloaded_count} images!")
            
        except Exception as e:
            self.log_message(f"Error during scraping: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        
        finally:
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.progress.stop()
            if self.is_scraping:
                self.status_var.set("Ready")
    
    def scrape_platforms(self, platforms, person_name):
        """Scrape images from specified platforms"""
        results = []
        
        for platform, username in platforms.items():
            if not self.is_scraping:
                break
            
            self.log_message(f"Checking {platform}: {username}")
            
            try:
                if platform == 'twitter':
                    img = self.scrape_twitter_profile(username)
                elif platform == 'github':
                    img = self.scrape_github_profile(username)
                elif platform == 'linkedin':
                    img = self.scrape_linkedin_profile(username)
                
                if img:
                    results.append(img)
                    self.log_message(f"Found image from {platform}")
            except Exception as e:
                self.log_message(f"Error scraping {platform}: {str(e)}")
            
            time.sleep(1)  # Rate limiting
        
        return results
    
    def download_and_process_image(self, img_info, person_name, current, total):
        """Download and process a single image with advanced validation"""
        try:
            if not self.is_scraping:
                return False
                
            self.log_message(f"Processing image {current}/{total} from {img_info['source']}")
            filepath = self.download_image(img_info, person_name)
            
            if filepath:
                # Additional validation for downloaded image
                if self.face_detection_enabled:
                    try:
                        # Load the image
                        image = cv2.imread(filepath)
                        if image is None:
                            raise ValueError("Could not read downloaded image")
                        
                        # Convert to RGB
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Detect faces
                        faces = self.face_detector(rgb_image, 1)
                        if len(faces) == 0:
                            os.remove(filepath)
                            self.log_message(f"✗ No faces in downloaded image: {os.path.basename(filepath)}")
                            return False
                            
                        # Check face sizes
                        img_height, img_width = image.shape[:2]
                        min_dim = min(img_height, img_width)
                        
                        valid_face_found = False
                        for face in faces:
                            if (face.width() >= self.min_face_size and 
                                face.height() >= self.min_face_size):
                                valid_face_found = True
                                break
                                
                        if not valid_face_found:
                            os.remove(filepath)
                            self.log_message(f"✗ Faces too small in: {os.path.basename(filepath)}")
                            return False
                            
                        # Face matching if enabled
                        if self.face_matching_enabled and self.reference_faces:
                            match_found = False
                            for face in faces:
                                shape = self.shape_predictor(rgb_image, face)
                                face_descriptor = self.face_recognizer.compute_face_descriptor(rgb_image, shape)
                                face_encoding = np.array(face_descriptor)
                                
                                for ref_face in self.reference_faces:
                                    similarity = cosine_similarity(
                                        [face_encoding],
                                        [ref_face['encoding']]
                                    )[0][0]
                                    
                                    if similarity >= self.similarity_threshold:
                                        match_found = True
                                        break
                                
                                if match_found:
                                    break
                            
                            if not match_found:
                                os.remove(filepath)
                                self.log_message(f"✗ No face match in: {os.path.basename(filepath)}")
                                return False
                    
                    except Exception as e:
                        self.log_message(f"Error processing downloaded image: {str(e)}")
                        return False
                
                # Update preview
                self.update_image_preview(filepath)
                
                self.log_message(f"✓ Success: {os.path.basename(filepath)}")
                return True
            else:
                self.log_message(f"✗ Failed to download from {img_info['source']}")
                return False
                
        except Exception as e:
            self.log_message(f"Error processing image: {str(e)}")
            return False
    
    def update_image_preview(self, filepath):
        """Update the image preview panel"""
        try:
            img = Image.open(filepath)
            img.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(img)
            
            self.preview_label.config(image=photo)
            self.preview_label.image = photo  # Keep reference
            self.preview_label.config(text="")
            
        except Exception as e:
            self.preview_label.config(text="Preview unavailable")
            self.log_message(f"Preview error: {str(e)}")
    
    def search_google_images(self, query, max_results):
        """Enhanced Google Images search with multiple attempts"""
        # [Implementation similar to previous version but with improved error handling]
        pass
    
    def search_alternative_sources(self, query, max_results):
        """Search alternative image sources"""
        # [Implementation similar to previous version but with improved error handling]
        pass
    
    def scrape_twitter_profile(self, username):
        """Enhanced Twitter profile scraping"""
        # [Implementation similar to previous version but with improved error handling]
        pass
    
    def scrape_github_profile(self, username):
        """Enhanced GitHub profile scraping"""
        # [Implementation similar to previous version but with improved error handling]
        pass
    
    def scrape_linkedin_profile(self, profile_url):
        """Enhanced LinkedIn profile scraping"""
        # [Implementation similar to previous version but with improved error handling]
        pass
    
    def download_image(self, image_info, person_name):
        """Download an image with improved error handling"""
        try:
            response = self.session.get(image_info['url'], stream=True, timeout=self.timeout_var.get())
            response.raise_for_status()
            
            # Generate filename
            url_hash = hashlib.md5(image_info['url'].encode()).hexdigest()[:8]
            source = image_info['source'].replace('/', '_').replace(' ', '_')
            person_folder = os.path.join(self.download_folder, person_name.replace(' ', '_'))
            os.makedirs(person_folder, exist_ok=True)
            
            # Determine file extension from content type
            content_type = response.headers.get('content-type', '')
            if 'jpeg' in content_type or 'jpg' in content_type:
                ext = '.jpg'
            elif 'png' in content_type:
                ext = '.png'
            elif 'gif' in content_type:
                ext = '.gif'
            elif 'webp' in content_type:
                ext = '.webp'
            else:
                # Try to determine from URL
                if '.jpg' in image_info['url'].lower():
                    ext = '.jpg'
                elif '.png' in image_info['url'].lower():
                    ext = '.png'
                elif '.gif' in image_info['url'].lower():
                    ext = '.gif'
                else:
                    ext = '.jpg'
            
            filename = f"{source}_{url_hash}{ext}"
            filepath = os.path.join(person_folder, filename)
            
            # Download and save
            with open(filepath, 'wb') as
