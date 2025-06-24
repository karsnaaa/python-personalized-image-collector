import streamlit as st
import requests
import os
import time
import threading
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import json
from typing import List, Dict, Optional
import hashlib
from PIL import Image, ImageDraw
import io
import cv2
import numpy as np
from pathlib import Path
import face_recognition
import base64
from datetime import datetime
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModernProfileImageCollector:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Initialize face detection
        try:
            # Load OpenCV face cascade (more reliable than face_recognition for detection)
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        except:
            st.error("Face detection not available. Install opencv-python for better results.")
            self.face_cascade = None
            self.profile_cascade = None
    
    def has_face(self, image_bytes: bytes) -> tuple[bool, int]:
        """
        Check if image contains a face and return confidence score
        Returns: (has_face: bool, face_count: int)
        """
        try:
            # Convert bytes to opencv image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return False, 0
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            face_count = 0
            
            # Detect frontal faces
            if self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                face_count += len(faces)
            
            # Detect profile faces
            if self.profile_cascade is not None:
                profile_faces = self.profile_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                face_count += len(profile_faces)
            
            # Additional check: image should be reasonably sized for a profile pic
            height, width = img.shape[:2]
            min_size = 100  # Minimum 100x100 pixels
            
            if width < min_size or height < min_size:
                return False, 0
            
            return face_count > 0, face_count
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return False, 0
    
    def is_profile_image_quality(self, image_bytes: bytes) -> dict:
        """
        Analyze image quality for profile picture suitability
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))
            width, height = img.size
            
            # Calculate quality metrics
            aspect_ratio = width / height
            pixel_count = width * height
            
            quality_score = 0
            reasons = []
            
            # Size scoring
            if pixel_count >= 250000:  # ~500x500 or better
                quality_score += 3
                reasons.append("High resolution")
            elif pixel_count >= 40000:  # ~200x200 or better
                quality_score += 2
                reasons.append("Good resolution")
            elif pixel_count >= 10000:  # ~100x100 or better
                quality_score += 1
                reasons.append("Acceptable resolution")
            else:
                reasons.append("Low resolution")
            
            # Aspect ratio scoring (prefer square-ish images)
            if 0.8 <= aspect_ratio <= 1.25:
                quality_score += 2
                reasons.append("Good aspect ratio")
            elif 0.6 <= aspect_ratio <= 1.6:
                quality_score += 1
                reasons.append("Acceptable aspect ratio")
            else:
                reasons.append("Poor aspect ratio")
            
            # File size check
            file_size = len(image_bytes)
            if file_size > 50000:  # > 50KB
                quality_score += 1
                reasons.append("Good file size")
            elif file_size < 5000:  # < 5KB (likely too small)
                quality_score -= 1
                reasons.append("File too small")
            
            return {
                'score': quality_score,
                'reasons': reasons,
                'dimensions': f"{width}x{height}",
                'file_size': f"{file_size/1024:.1f}KB",
                'aspect_ratio': round(aspect_ratio, 2)
            }
            
        except Exception as e:
            return {'score': 0, 'reasons': [f"Error analyzing: {e}"], 'dimensions': 'Unknown', 'file_size': 'Unknown'}
    
    def search_social_platforms(self, person_name: str, platforms: dict) -> List[Dict]:
        """Enhanced social platform searching"""
        images = []
        
        for platform, username in platforms.items():
            if not username.strip():
                continue
                
            st.write(f"🔍 Searching {platform.title()} for: {username}")
            
            try:
                if platform == 'github':
                    img = self.get_github_profile(username)
                elif platform == 'twitter':
                    img = self.get_twitter_profile(username)
                elif platform == 'linkedin':
                    img = self.get_linkedin_profile(username)
                elif platform == 'instagram':
                    img = self.get_instagram_profile(username)
                
                if img:
                    images.append(img)
                    st.success(f"✅ Found profile image from {platform.title()}")
                else:
                    st.warning(f"⚠️ No image found on {platform.title()}")
                    
            except Exception as e:
                st.error(f"❌ Error searching {platform}: {e}")
                
            time.sleep(1)  # Rate limiting
        
        return images
    
    def get_github_profile(self, username: str) -> Optional[Dict]:
        """Get GitHub profile with enhanced data"""
        try:
            api_url = f"https://api.github.com/users/{username}"
            response = self.session.get(api_url, timeout=10)
            
            if response.status_code == 200:
                user_data = response.json()
                avatar_url = user_data.get('avatar_url')
                
                if avatar_url:
                    # Get higher resolution version
                    avatar_url = avatar_url.replace('?v=4', '?v=4&s=400')
                    
                    return {
                        'url': avatar_url,
                        'source': 'GitHub',
                        'username': username,
                        'name': user_data.get('name', ''),
                        'bio': user_data.get('bio', ''),
                        'followers': user_data.get('followers', 0),
                        'confidence': 'high'  # GitHub avatars are usually profile pics
                    }
        except Exception as e:
            logger.error(f"GitHub API error: {e}")
        
        return None
    
    def get_twitter_profile(self, username: str) -> Optional[Dict]:
        """Enhanced Twitter profile image extraction"""
        try:
            # Try both twitter.com and x.com
            urls = [f"https://twitter.com/{username}", f"https://x.com/{username}"]
            
            for url in urls:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Multiple methods to find profile image
                    methods = [
                        ('meta[property="og:image"]', 'content'),
                        ('meta[name="twitter:image"]', 'content'),
                        ('img[src*="profile_images"]', 'src'),
                        ('img[alt*="profile"]', 'src')
                    ]
                    
                    for selector, attr in methods:
                        element = soup.select_one(selector)
                        if element and element.get(attr):
                            img_url = element.get(attr)
                            
                            # Skip default images
                            if 'default_profile' not in img_url and 'twimg.com' in img_url:
                                # Get higher quality version
                                img_url = img_url.replace('_normal', '_400x400')
                                img_url = img_url.replace('_bigger', '_400x400')
                                
                                return {
                                    'url': img_url,
                                    'source': 'Twitter/X',
                                    'username': username,
                                    'confidence': 'high',
                                    'method': selector
                                }
        except Exception as e:
            logger.error(f"Twitter error: {e}")
        
        return None
    
    def get_linkedin_profile(self, profile_url: str) -> Optional[Dict]:
        """Get LinkedIn profile image"""
        try:
            response = self.session.get(profile_url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # LinkedIn uses og:image for profile pictures
                og_image = soup.select_one('meta[property="og:image"]')
                if og_image and og_image.get('content'):
                    img_url = og_image.get('content')
                    return {
                        'url': img_url,
                        'source': 'LinkedIn',
                        'profile_url': profile_url,
                        'confidence': 'high'
                    }
        except Exception as e:
            logger.error(f"LinkedIn error: {e}")
        
        return None
    
    def get_instagram_profile(self, username: str) -> Optional[Dict]:
        """Attempt to get Instagram profile (limited due to restrictions)"""
        try:
            url = f"https://www.instagram.com/{username}/"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for profile image in meta tags
                og_image = soup.select_one('meta[property="og:image"]')
                if og_image and og_image.get('content'):
                    return {
                        'url': og_image.get('content'),
                        'source': 'Instagram',
                        'username': username,
                        'confidence': 'medium'
                    }
        except Exception as e:
            logger.error(f"Instagram error: {e}")
        
        return None
    
    def search_web_images(self, person_name: str, additional_terms: List[str] = None) -> List[Dict]:
        """Enhanced web image search with face validation"""
        images = []
        
        # Create targeted search queries
        base_terms = ["profile", "headshot", "photo", "picture", "portrait"]
        if additional_terms:
            base_terms.extend(additional_terms)
        
        search_queries = [
            f'"{person_name}" {term}' for term in base_terms[:3]
        ]
        
        for query in search_queries:
            st.write(f"🌐 Searching web for: {query}")
            
            try:
                # DuckDuckGo search (more reliable than Google for scraping)
                search_url = f"https://duckduckgo.com/?q={query.replace(' ', '+')}&iax=images&ia=images"
                
                response = self.session.get(search_url, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract image URLs from various sources
                    img_urls = self.extract_image_urls_from_page(soup, person_name)
                    
                    for img_url in img_urls[:5]:  # Limit per query
                        if img_url not in [img['url'] for img in images]:
                            images.append({
                                'url': img_url,
                                'source': 'Web Search',
                                'query': query,
                                'confidence': 'medium'
                            })
                
                time.sleep(2)  # Respectful delay
                
            except Exception as e:
                logger.error(f"Web search error for {query}: {e}")
        
        return images
    
    def extract_image_urls_from_page(self, soup: BeautifulSoup, person_name: str) -> List[str]:
        """Extract potential profile image URLs from a page"""
        urls = []
        
        # Look for various image sources
        selectors = [
            'img[src*="profile"]',
            'img[src*="avatar"]',
            'img[src*="headshot"]',
            'img[alt*="{}"]'.format(person_name.split()[0].lower()),
            'img[title*="{}"]'.format(person_name.split()[0].lower()),
        ]
        
        for selector in selectors:
            imgs = soup.select(selector)
            for img in imgs:
                src = img.get('src') or img.get('data-src')
                if src and src.startswith('http'):
                    # Filter out obviously non-profile images
                    if not any(skip in src.lower() for skip in ['logo', 'icon', 'banner', 'ad', 'button']):
                        urls.append(src)
        
        # Also check script tags for JSON data containing image URLs
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string:
                content = str(script.string)
                if 'http' in content and any(ext in content for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                    import re
                    found_urls = re.findall(r'https://[^"\s]+\.(?:jpg|jpeg|png|webp)', content, re.IGNORECASE)
                    urls.extend(found_urls[:3])  # Limit to avoid spam
        
        return list(set(urls))  # Remove duplicates
    
    def validate_and_score_images(self, images: List[Dict]) -> List[Dict]:
        """Validate images and score them based on face detection and quality"""
        validated_images = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, img_info in enumerate(images):
            progress = (i + 1) / len(images)
            progress_bar.progress(progress)
            status_text.text(f"Validating image {i+1}/{len(images)} from {img_info['source']}")
            
            try:
                # Download image for validation
                response = self.session.get(img_info['url'], timeout=10)
                if response.status_code == 200:
                    image_bytes = response.content
                    
                    # Check for face
                    has_face, face_count = self.has_face(image_bytes)
                    
                    if has_face:
                        # Analyze quality
                        quality_info = self.is_profile_image_quality(image_bytes)
                        
                        # Calculate overall score
                        confidence_multiplier = {'high': 3, 'medium': 2, 'low': 1}
                        base_score = confidence_multiplier.get(img_info.get('confidence', 'medium'), 2)
                        face_score = min(face_count * 2, 4)  # Max 4 points for faces
                        quality_score = quality_info['score']
                        
                        total_score = base_score + face_score + quality_score
                        
                        img_info.update({
                            'has_face': True,
                            'face_count': face_count,
                            'quality_info': quality_info,
                            'total_score': total_score,
                            'image_bytes': image_bytes
                        })
                        
                        validated_images.append(img_info)
                        
                    else:
                        st.write(f"❌ No face detected in image from {img_info['source']}")
                
            except Exception as e:
                logger.error(f"Error validating image from {img_info['source']}: {e}")
        
        progress_bar.empty()
        status_text.empty()
        
        # Sort by score (highest first)
        validated_images.sort(key=lambda x: x.get('total_score', 0), reverse=True)
        
        return validated_images
    
    def save_images(self, images: List[Dict], person_name: str, output_dir: str) -> List[str]:
        """Save validated images to disk"""
        saved_files = []
        person_folder = Path(output_dir) / person_name.replace(' ', '_')
        person_folder.mkdir(parents=True, exist_ok=True)
        
        for i, img_info in enumerate(images):
            try:
                # Generate filename
                source = img_info['source'].replace('/', '_').replace(' ', '_')
                score = img_info.get('total_score', 0)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Determine extension
                content_type = self.session.head(img_info['url']).headers.get('content-type', '')
                if 'jpeg' in content_type or 'jpg' in content_type:
                    ext = '.jpg'
                elif 'png' in content_type:
                    ext = '.png'
                else:
                    ext = '.jpg'
                
                filename = f"{source}_score{score:.1f}_{timestamp}{ext}"
                filepath = person_folder / filename
                
                # Save image
                with open(filepath, 'wb') as f:
                    f.write(img_info['image_bytes'])
                
                saved_files.append(str(filepath))
                
            except Exception as e:
                logger.error(f"Error saving image: {e}")
        
        return saved_files

def main():
    st.set_page_config(
        page_title="Profile Image Collector",
        page_icon="📸",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">📸 AI-Powered Profile Image Collector</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Person information
        person_name = st.text_input("👤 Person Name", placeholder="Enter full name")
        
        st.subheader("🌐 Social Media Accounts")
        social_platforms = {}
        social_platforms['github'] = st.text_input("GitHub Username", placeholder="username")
        social_platforms['twitter'] = st.text_input("Twitter/X Username", placeholder="username")
        social_platforms['linkedin'] = st.text_input("LinkedIn Profile URL", placeholder="https://linkedin.com/in/username")
        social_platforms['instagram'] = st.text_input("Instagram Username", placeholder="username")
        
        st.subheader("⚙️ Settings")
        max_images = st.slider("Maximum Images to Collect", 1, 20, 10)
        output_dir = st.text_input("Output Directory", value="profile_images")
        
        # Additional search terms
        additional_terms = st.text_area("Additional Search Terms (one per line)", 
                                      placeholder="CEO\nfounder\nauthor")
        additional_terms_list = [term.strip() for term in additional_terms.split('\n') if term.strip()]
        
        face_detection_enabled = st.checkbox("Enable Face Detection", value=True)
        
    # Main content area
    if not person_name:
        st.markdown("""
        <div class="info-box">
        <h3>🚀 Welcome to the Profile Image Collector!</h3>
        <p>This tool helps you collect high-quality profile images of specific people using:</p>
        <ul>
        <li>🤖 AI-powered face detection</li>
        <li>🔍 Multi-platform social media search</li>
        <li>📊 Image quality analysis</li>
        <li>🎯 Targeted web search</li>
        </ul>
        <p><strong>Get started by entering a person's name in the sidebar!</strong></p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Initialize collector
    if 'collector' not in st.session_state:
        st.session_state.collector = ModernProfileImageCollector()
    
    collector = st.session_state.collector
    
    # Start collection button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Collection", type="primary", use_container_width=True):
            st.session_state.collecting = True
            st.session_state.results = None
    
    # Collection process
    if st.session_state.get('collecting', False):
        st.markdown("---")
        st.header(f"🔍 Collecting Profile Images for: {person_name}")
        
        all_images = []
        
        # Step 1: Social media platforms
        if any(social_platforms.values()):
            st.subheader("1️⃣ Searching Social Media Platforms")
            social_images = collector.search_social_platforms(person_name, social_platforms)
            all_images.extend(social_images)
            st.write(f"Found {len(social_images)} images from social platforms")
        
        # Step 2: Web search
        st.subheader("2️⃣ Searching the Web")
        web_images = collector.search_web_images(person_name, additional_terms_list)
        all_images.extend(web_images)
        st.write(f"Found {len(web_images)} additional images from web search")
        
        # Step 3: Validation and scoring
        if all_images:
            st.subheader("3️⃣ Validating Images with Face Detection")
            if face_detection_enabled:
                validated_images = collector.validate_and_score_images(all_images)
            else:
                validated_images = all_images
                for img in validated_images:
                    img['has_face'] = True  # Skip validation
                    img['total_score'] = 5  # Default score
            
            # Limit to max_images
            final_images = validated_images[:max_images]
            
            if final_images:
                st.subheader("4️⃣ Saving Images")
                saved_files = collector.save_images(final_images, person_name, output_dir)
                
                st.session_state.results = {
                    'person_name': person_name,
                    'images': final_images,
                    'saved_files': saved_files
                }
                
                st.success(f"✅ Successfully collected and saved {len(saved_files)} profile images!")
            else:
                st.error("❌ No valid profile images found with faces detected.")
        else:
            st.error("❌ No images found. Try different search terms or social media accounts.")
        
        st.session_state.collecting = False
    
    # Display results
    if st.session_state.get('results'):
        results = st.session_state.results
        st.markdown("---")
        st.header("📊 Collection Results")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Images Found", len(results['images']))
        with col2:
            st.metric("Images Saved", len(results['saved_files']))
        with col3:
            avg_score = sum(img.get('total_score', 0) for img in results['images']) / len(results['images'])
            st.metric("Avg Quality Score", f"{avg_score:.1f}")
        with col4:
            face_count = sum(img.get('face_count', 0) for img in results['images'])
            st.metric("Total Faces Detected", face_count)
        
        # Display images
        st.subheader("🖼️ Collected Images")
        
        for i, img_info in enumerate(results['images']):
            with st.expander(f"Image {i+1} - {img_info['source']} (Score: {img_info.get('total_score', 0):.1f})"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if 'image_bytes' in img_info:
                        image = Image.open(io.BytesIO(img_info['image_bytes']))
                        st.image(image, caption=f"From {img_info['source']}", use_column_width=True)
                
                with col2:
                    st.write(f"**Source:** {img_info['source']}")
                    st.write(f"**URL:** {img_info['url']}")
                    if 'face_count' in img_info:
                        st.write(f"**Faces Detected:** {img_info['face_count']}")
                    if 'quality_info' in img_info:
                        quality = img_info['quality_info']
                        st.write(f"**Dimensions:** {quality['dimensions']}")
                        st.write(f"**File Size:** {quality['file_size']}")
                        st.write(f"**Quality Reasons:** {', '.join(quality['reasons'])}")
        
        # Download button for results
        if st.button("📁 Open Output Folder", use_container_width=True):
            folder_path = Path(output_dir) / results['person_name'].replace(' ', '_')
            st.write(f"Images saved to: `{folder_path.absolute()}`")

if __name__ == "__main__":
    main()