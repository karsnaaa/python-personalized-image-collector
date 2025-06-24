import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import requests
import os
import time
import threading
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import json
from typing import List, Dict, Optional
import hashlib
from PIL import Image, ImageTk
import io
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import face_recognition

class ProfileImageScraperGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Profile Image Scraper with Face Recognition")
        self.root.geometry("1000x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize scraper session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        self.download_folder = "profile_images"
        self.is_scraping = False
        self.face_detection_enabled = True
        self.reference_faces = []
        
        self.create_widgets()
    
    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="AI Profile Image Scraper with Face Recognition", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0', fg='#333')
        title_label.pack(pady=10)
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Person name input
        name_frame = ttk.Frame(main_frame)
        name_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(name_frame, text="Person Name:", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        self.name_entry = ttk.Entry(name_frame, font=('Arial', 11), width=50)
        self.name_entry.pack(fill=tk.X, pady=(5, 0))
        
        # Reference image frame
        ref_frame = ttk.LabelFrame(main_frame, text="Reference Images (For Face Matching)", padding="10")
        ref_frame.pack(fill=tk.X, pady=(0, 15))
        
        ref_buttons_frame = ttk.Frame(ref_frame)
        ref_buttons_frame.pack(fill=tk.X)
        
        ttk.Button(ref_buttons_frame, text="Add Reference Image", command=self.add_reference_image).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(ref_buttons_frame, text="Clear References", command=self.clear_reference_images).pack(side=tk.LEFT)
        
        self.ref_images_frame = ttk.Frame(ref_frame)
        self.ref_imagesFrame.pack(fill=tk.X)
        
        # Platforms frame
        platforms_frame = ttk.LabelFrame(main_frame, text="Platform Usernames (Optional)", padding="10")
        platforms_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Twitter
        twitter_frame = ttk.Frame(platforms_frame)
        twitter_frame.pack(fill=tk.X, pady=2)
        ttk.Label(twitter_frame, text="Twitter/X:", width=12).pack(side=tk.LEFT)
        self.twitter_entry = ttk.Entry(twitter_frame, width=30)
        self.twitter_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # GitHub
        github_frame = ttk.Frame(platforms_frame)
        github_frame.pack(fill=tk.X, pady=2)
        ttk.Label(github_frame, text="GitHub:", width=12).pack(side=tk.LEFT)
        self.github_entry = ttk.Entry(github_frame, width=30)
        self.github_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # LinkedIn
        linkedin_frame = ttk.Frame(platforms_frame)
        linkedin_frame.pack(fill=tk.X, pady=2)
        ttk.Label(linkedin_frame, text="LinkedIn URL:", width=12).pack(side=tk.LEFT)
        self.linkedin_entry = ttk.Entry(linkedin_frame, width=40)
        self.linkedin_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # Settings frame
        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Face detection toggle
        face_detect_frame = ttk.Frame(settings_frame)
        face_detect_frame.pack(fill=tk.X, pady=2)
        ttk.Label(face_detect_frame, text="Face Detection:", width=15).pack(side=tk.LEFT)
        self.face_detect_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(face_detect_frame, variable=self.face_detect_var, 
                        command=self.toggle_face_detection).pack(side=tk.LEFT)
        
        # Folder selection
        folder_frame = ttk.Frame(settings_frame)
        folder_frame.pack(fill=tk.X, pady=2)
        ttk.Label(folder_frame, text="Download Folder:", width=15).pack(side=tk.LEFT)
        self.folder_var = tk.StringVar(value=self.download_folder)
        self.folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_var, width=35)
        self.folder_entry.pack(side=tk.LEFT, padx=(5, 5))
        ttk.Button(folder_frame, text="Browse", command=self.browse_folder).pack(side=tk.LEFT)
        
        # Max images setting
        max_frame = ttk.Frame(settings_frame)
        max_frame.pack(fill=tk.X, pady=2)
        ttk.Label(max_frame, text="Max Images:", width=15).pack(side=tk.LEFT)
        self.max_images_var = tk.StringVar(value="10")
        ttk.Entry(max_frame, textvariable=self.max_images_var, width=10).pack(side=tk.LEFT, padx=(5, 0))
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.start_button = ttk.Button(buttons_frame, text="Start Scraping", 
                                     command=self.start_scraping, style='Accent.TButton')
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(buttons_frame, text="Stop", 
                                    command=self.stop_scraping, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(buttons_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(buttons_frame, text="Open Folder", command=self.open_download_folder).pack(side=tk.LEFT)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(0, 10))
        
        # Log area
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def add_reference_image(self):
        """Add reference image for face matching"""
        file_path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            try:
                # Load the image and encode the face
                image = face_recognition.load_image_file(file_path)
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    self.reference_faces.append(encodings[0])
                    
                    # Display thumbnail in the reference frame
                    img = Image.open(file_path)
                    img.thumbnail((100, 100))
                    photo = ImageTk.PhotoImage(img)
                    
                    label = tk.Label(self.ref_images_frame, image=photo)
                    label.image = photo  # Keep a reference
                    label.pack(side=tk.LEFT, padx=5)
                    
                    self.log_message(f"Added reference image: {os.path.basename(file_path)}")
                else:
                    messagebox.showerror("Error", "No faces found in the selected image")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def clear_reference_images(self):
        """Clear all reference images"""
        self.reference_faces = []
        for widget in self.ref_images_frame.winfo_children():
            widget.destroy()
        self.log_message("Cleared all reference images")
    
    def toggle_face_detection(self):
        """Toggle face detection on/off"""
        self.face_detection_enabled = self.face_detect_var.get()
        status = "enabled" if self.face_detection_enabled else "disabled"
        self.log_message(f"Face detection {status}")
    
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
    
    def contains_face(self, image_data: bytes) -> bool:
        """Check if image contains at least one face"""
        try:
            # Convert bytes to numpy array
            image = face_recognition.load_image_file(io.BytesIO(image_data))
            
            # Find all face locations
            face_locations = face_recognition.face_locations(image)
            
            return len(face_locations) > 0
        except Exception as e:
            self.log_message(f"Face detection error: {str(e)}")
            return False
    
    def matches_reference_faces(self, image_data: bytes) -> bool:
        """Check if image contains a face that matches any reference faces"""
        if not self.reference_faces:
            return True  # No reference faces to compare against
            
        try:
            # Load the image
            unknown_image = face_recognition.load_image_file(io.BytesIO(image_data))
            
            # Get face encodings for the unknown image
            unknown_encodings = face_recognition.face_encodings(unknown_image)
            
            if not unknown_encodings:
                return False  # No faces found in the image
                
            # Compare against each reference face
            for ref_encoding in self.reference_faces:
                matches = face_recognition.compare_faces([ref_encoding], unknown_encodings[0], tolerance=0.6)
                if any(matches):
                    return True
                    
            return False
        except Exception as e:
            self.log_message(f"Face matching error: {str(e)}")
            return False
    
    def validate_image(self, url: str) -> bool:
        """Validate if URL is likely to be a real image with a face"""
        try:
            # First check if it's an image URL
            response = self.session.head(url, timeout=5)
            content_type = response.headers.get('content-type', '').lower()
            
            if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'jpg', 'png', 'gif', 'webp']):
                return False
                
            # If face detection is disabled, skip further checks
            if not self.face_detection_enabled:
                return True
                
            # Download the full image for face detection
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            image_data = response.content
            
            # Check if image contains at least one face
            if not self.contains_face(image_data):
                return False
                
            # If we have reference faces, check for matches
            if self.reference_faces and not self.matches_reference_faces(image_data):
                return False
                
            return True
            
        except Exception as e:
            self.log_message(f"Validation error for {url}: {str(e)}")
            return False
    
    def scrape_images(self, person_name):
        """Main scraping method (runs in separate thread)"""
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
            
            max_images = int(self.max_images_var.get() or 10)
            
            self.log_message(f"Starting scrape for: {person_name}")
            self.log_message(f"Max images: {max_images}")
            if self.face_detection_enabled:
                self.log_message("Face detection is ENABLED")
                if self.reference_faces:
                    self.log_message(f"Using {len(self.reference_faces)} reference faces for matching")
            else:
                self.log_message("Face detection is DISABLED")
            
            all_images = []
            
            # Check specific platforms
            for platform, username in platforms.items():
                if not self.is_scraping:
                    break
                
                self.log_message(f"Checking {platform}: {username}")
                
                if platform == 'twitter':
                    img = self.scrape_twitter_profile(username)
                elif platform == 'github':
                    img = self.scrape_github_profile(username)
                elif platform == 'linkedin':
                    img = self.scrape_linkedin_profile(username)
                
                if img:
                    all_images.append(img)
                    self.log_message(f"Found image from {platform}")
                
                time.sleep(1)  # Rate limiting
            
            # Search Google Images
            if self.is_scraping and len(all_images) < max_images:
                self.log_message("Searching Google Images...")
                google_images = self.search_google_images(person_name, max_images - len(all_images))
                all_images.extend(google_images)
                self.log_message(f"Found {len(google_images)} images from Google")
            
            # Search alternative sources if needed
            if self.is_scraping and len(all_images) < max_images:
                self.log_message("Searching alternative sources...")
                alt_images = self.search_alternative_sources(person_name, max_images - len(all_images))
                all_images.extend(alt_images)
                self.log_message(f"Found {len(alt_images)} images from alternative sources")
            
            # Validate and filter images
            valid_images = []
            for img in all_images:
                if not self.is_scraping:
                    break
                
                if self.validate_image(img['url']):
                    valid_images.append(img)
                    self.log_message(f"✓ Valid image from {img['source']}")
                else:
                    reason = "no face detected" if self.face_detection_enabled else "invalid/inaccessible"
                    self.log_message(f"✗ Rejected image from {img['source']} ({reason})")
            
            # Download images with threading for better performance
            downloaded_count = 0
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for i, img_info in enumerate(valid_images[:max_images]):
                    if not self.is_scraping:
                        break
                    
                    futures.append(executor.submit(self.download_and_process_image, img_info, person_name, i+1, len(valid_images)))
                
                for future in futures:
                    result = future.result()
                    if result:
                        downloaded_count += 1
            
            if self.is_scraping:
                self.log_message(f"Scraping completed! Downloaded {downed_count} images.")
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
    
    def download_and_process_image(self, img_info, person_name, current, total):
        """Download and process a single image with face detection"""
        try:
            if not self.is_scraping:
                return False
                
            self.log_message(f"Downloading image {current}/{total} from {img_info['source']}")
            filepath = self.download_image(img_info, person_name)
            
            if filepath:
                # Verify the downloaded image contains a face if face detection is enabled
                if self.face_detection_enabled:
                    try:
                        image = face_recognition.load_image_file(filepath)
                        face_locations = face_recognition.face_locations(image)
                        
                        if not face_locations:
                            os.remove(filepath)
                            self.log_message(f"✗ No face detected in downloaded image: {os.path.basename(filepath)}")
                            return False
                            
                        # If we have reference faces, check for matches
                        if self.reference_faces:
                            face_encodings = face_recognition.face_encodings(image, face_locations)
                            match_found = False
                            
                            for face_encoding in face_encodings:
                                matches = face_recognition.compare_faces(self.reference_faces, face_encoding, tolerance=0.6)
                                if any(matches):
                                    match_found = True
                                    break
                            
                            if not match_found:
                                os.remove(filepath)
                                self.log_message(f"✗ No matching face in downloaded image: {os.path.basename(filepath)}")
                                return False
                    
                    except Exception as e:
                        self.log_message(f"Error processing downloaded image: {str(e)}")
                        return False
                
                self.log_message(f"✓ Downloaded: {os.path.basename(filepath)}")
                return True
            else:
                self.log_message(f"✗ Failed to download from {img_info['source']}")
                return False
                
        except Exception as e:
            self.log_message(f"Error downloading image: {str(e)}")
            return False
    
    # [Rest of your existing methods (search_alternative_sources, validate_image_url, 
    # search_google_images, scrape_github_profile, scrape_twitter_profile, 
    # scrape_linkedin_profile, download_image) remain unchanged...]

def main():
    root = tk.Tk()
    app = ProfileImageScraperGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
