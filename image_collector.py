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
import re

class AdvancedProfileImageScraper:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced AI Profile Image Scraper")
        self.root.geometry("1100x900")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize face detection models
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
        
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
        self.min_face_size = 100
        self.similarity_threshold = 0.7
        self.name_similarity_threshold = 0.8  # Threshold for name matching in metadata
        
        self.create_widgets()
        self.setup_advanced_settings()

    # ... [Previous methods remain the same until validate_image] ...

    def validate_image(self, url: str, context: dict = None) -> bool:
        """Enhanced image validation with multiple checks including relevance"""
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
            
            # Download the full image for further validation
            response = self.session.get(url, timeout=self.timeout_var.get())
            response.raise_for_status()
            
            image_data = response.content
            image = np.array(Image.open(io.BytesIO(image_data)))
            
            # Convert to RGB (dlib uses RGB)
            if len(image.shape) == 2:  # Grayscale
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Check for faces if enabled
            if self.face_detection_enabled:
                faces = self.face_detector(rgb_image, 1)
                if len(faces) == 0:
                    return False
                
                # Check face sizes
                img_height, img_width = rgb_image.shape[:2]
                min_dim = min(img_height, img_width)
                
                valid_face_found = False
                for face in faces:
                    face_area = face.width() * face.height()
                    min_area = self.min_face_size ** 2
                    
                    if face_area >= min_area:
                        valid_face_found = True
                        break
                
                if not valid_face_found:
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
                        return False
            
            # Check image metadata for relevance
            if context and 'person_name' in context:
                person_name = context['person_name'].lower()
                
                # Try to extract metadata from URL
                url_path = urlparse(url).path.lower()
                filename = os.path.basename(url_path)
                
                # Check if person's name appears in URL or filename
                name_parts = person_name.split()
                name_in_url = any(part in url_path for part in name_parts) or any(part in filename for part in name_parts)
                
                if not name_in_url:
                    # Check if this is a profile picture (common patterns)
                    is_profile_pic = any(term in url_path for term in ['profile', 'avatar', 'picture', 'photo', 'portrait', 'user'])
                    if not is_profile_pic:
                        return False
            
            return True
            
        except Exception as e:
            self.log_message(f"Validation error for {url}: {str(e)}")
            return False

    def scrape_images(self, person_name):
        """Main scraping method with enhanced relevance filtering"""
        try:
            self.download_folder = self.folder_var.get()
            os.makedirs(self.download_folder, exist_ok=True)
            
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
            
            all_images = []
            
            # Check specific platforms first (most relevant)
            platform_results = self.scrape_platforms(platforms, person_name)
            all_images.extend(platform_results)
            
            # Search Google Images with relevance filters
            if self.is_scraping and len(all_images) < max_images:
                google_images = self.search_google_images(person_name, max_images - len(all_images))
                all_images.extend(google_images)
            
            # Validate and filter images for relevance
            valid_images = []
            for img in all_images:
                if not self.is_scraping:
                    break
                
                # Add context for relevance checking
                context = {
                    'person_name': person_name,
                    'source': img.get('source', 'unknown')
                }
                
                if self.validate_image(img['url'], context):
                    valid_images.append(img)
                    self.log_message(f"✓ Relevant image from {img['source']}")
                else:
                    self.log_message(f"✗ Irrelevant image from {img['source']}")
            
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
                self.log_message(f"Scraping completed! Downloaded {downloaded_count} relevant images.")
                self.status_var.set(f"Completed - {downloaded_count} relevant images downloaded")
                messagebox.showinfo("Complete", f"Successfully downloaded {downloaded_count} relevant images!")
            
        except Exception as e:
            self.log_message(f"Error during scraping: {str(e)}")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        
        finally:
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.progress.stop()
            if self.is_scraping:
                self.status_var.set("Ready")

    def search_google_images(self, query, max_results):
        """Enhanced Google Images search with relevance filters"""
        try:
            self.log_message(f"Searching Google Images for: {query}")
            
            # Create search URL with parameters to get more relevant results
            search_url = f"https://www.google.com/search?q={query}+profile+photo+OR+avatar+OR+portrait&tbm=isch"
            
            response = self.session.get(search_url, timeout=self.timeout_var.get())
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            images = []
            
            # Extract image URLs with additional relevance checks
            for img in soup.find_all('img'):
                img_url = img.get('src') or img.get('data-src')
                if not img_url or img_url.startswith('data:'):
                    continue
                
                # Check if URL looks like a profile image
                if any(term in img_url.lower() for term in ['profile', 'avatar', 'portrait', 'user']):
                    images.append({
                        'url': img_url,
                        'source': 'Google Images',
                        'relevance': 1.0  # High relevance
                    })
                elif 'http' in img_url:
                    images.append({
                        'url': img_url,
                        'source': 'Google Images',
                        'relevance': 0.5  # Medium relevance
                    })
            
            # Sort by relevance (highest first)
            images.sort(key=lambda x: x['relevance'], reverse=True)
            
            return images[:max_results]
            
        except Exception as e:
            self.log_message(f"Google Images search error: {str(e)}")
            return []

    def scrape_twitter_profile(self, username):
        """Enhanced Twitter profile scraping with relevance checks"""
        try:
            profile_url = f"https://twitter.com/{username}"
            response = self.session.get(profile_url, timeout=self.timeout_var.get())
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find profile image
            profile_image = soup.find('img', {'alt': f'Profile image for {username}'})
            if not profile_image:
                profile_image = soup.find('img', {'alt': 'Profile image'})
            
            if profile_image and profile_image.get('src'):
                return {
                    'url': profile_image['src'].replace('_normal', '_400x400'),  # Get higher resolution
                    'source': 'Twitter',
                    'relevance': 1.0  # Definitely relevant
                }
            
            return None
            
        except Exception as e:
            self.log_message(f"Twitter scrape error: {str(e)}")
            return None

    # ... [Other platform-specific methods with similar relevance enhancements] ...

    def download_image(self, image_info, person_name):
        """Download an image with additional relevance checks"""
        try:
            response = self.session.get(image_info['url'], stream=True, timeout=self.timeout_var.get())
            response.raise_for_status()
            
            # Generate filename with relevance indicator
            url_hash = hashlib.md5(image_info['url'].encode()).hexdigest()[:8]
            source = image_info['source'].replace('/', '_').replace(' ', '_')
            person_folder = os.path.join(self.download_folder, person_name.replace(' ', '_'))
            os.makedirs(person_folder, exist_ok=True)
            
            # Determine file extension
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
                ext = '.jpg'
            
            # Include relevance in filename
            relevance = image_info.get('relevance', 0.5)
            rel_tag = f"_rel{int(relevance*10)}"  # 0-10 scale
            
            filename = f"{source}_{url_hash}{rel_tag}{ext}"
            filepath = os.path.join(person_folder, filename)
            
            # Download and save
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            
            return filepath
            
        except Exception as e:
            self.log_message(f"Download error for {image_info['url']}: {str(e)}")
            return None
