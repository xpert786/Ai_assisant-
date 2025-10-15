# from django.test import TestCase

# # Create your tests here.







# import os
# import json
# from PIL import Image
# import re
# from django.conf import settings
# import logging
# import base64
# import openai
# import io
# import pkg_resources
# import requests
# import time
# import hashlib
# import numpy as np

# # Optional heavy deps
# try:
#     import cv2  # type: ignore
# except Exception:  # pragma: no cover
#     cv2 = None

# try:
#     import easyocr  # type: ignore
# except Exception:  # pragma: no cover
#     easyocr = None

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Check OpenAI version
# try:
#     openai_version = pkg_resources.get_distribution("openai").version
#     USING_OPENAI_V1 = int(openai_version.split('.')[0]) >= 1
# except Exception:
#     openai_version = 'unknown'
#     USING_OPENAI_V1 = False

# logger.info(f"Using OpenAI version: {openai_version}, V1 API: {USING_OPENAI_V1}")

# # Configure pytesseract path for Windows
# if os.name == 'nt':  # Windows
#     pass # No longer needed for OCR

# class AIClient:
#     """Client for interacting with OpenAI API"""
    
#     def __init__(self):
#         self.api_key = settings.OPENAI_API_KEY if hasattr(settings, 'OPENAI_API_KEY') else os.environ.get('OPENAI_API_KEY', '')
#         self.model = "gpt-4o"  # Using the latest model instead of deprecated gpt-4-vision-preview
        
#         if USING_OPENAI_V1:
#             self.client = openai.OpenAI(api_key=self.api_key)
#         else:
#             openai.api_key = self.api_key
            
#         self.airtable_client = AirtableClient()

#         # Initialize EasyOCR reader lazily to avoid heavy startup if not used
#         self._easyocr_reader = None
#         if easyocr is not None:
#             try:
#                 # Only English for now; extend if needed
#                 self._easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
#             except Exception as e:
#                 logger.error(f"Failed to initialize EasyOCR reader: {e}")
#                 self._easyocr_reader = None
        
#     def extract_client_info(self, text):
#         """Extract client name and contact info from text"""
#         if not text:
#             return None, None
            
#         try:
#             # Use OpenAI to extract client information
#             if USING_OPENAI_V1:
#                 response = self.client.chat.completions.create(
#                     model="gpt-4o",
#                     messages=[
#                         {"role": "system", "content": "You are a helpful assistant that extracts client information from text. Extract the name and contact information (email or phone) if present. Return in JSON format with keys 'name' and 'contact'."},
#                         {"role": "user", "content": text}
#                     ],
#                     temperature=0.3,
#                     max_tokens=150
#                 )
#                 content = response.choices[0].message.content.strip()
#             else:
#                 response = openai.ChatCompletion.create(
#                     model="gpt-4o",
#                     messages=[
#                         {"role": "system", "content": "You are a helpful assistant that extracts client information from text. Extract the name and contact information (email or phone) if present. Return in JSON format with keys 'name' and 'contact'."},
#                         {"role": "user", "content": text}
#                     ],
#                     temperature=0.3,
#                     max_tokens=150
#                 )
#                 content = response.choices[0].message.content.strip()
            
#             # Try to parse JSON from the response
#             try:
#                 data = json.loads(content)
#                 name = data.get('name')
#                 contact = data.get('contact')
                
#                 # Validate extracted information
#                 if name and len(name) > 2 and contact and len(contact) > 5:
#                     logger.info(f"Extracted client info: {name}, {contact}")
#                     return name, contact
#             except json.JSONDecodeError:
#                 # If not valid JSON, try regex extraction
#                 pass
                
#             # Fallback to regex extraction
#             # Look for email
#             email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
#             email_match = re.search(email_pattern, text)
            
#             # Look for phone number (simple pattern)
#             phone_pattern = r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b'
#             phone_match = re.search(phone_pattern, text)
            
#             # Look for name (this is a simple heuristic)
#             name_pattern = r'(?:I am|my name is|this is) ([A-Z][a-z]+(?: [A-Z][a-z]+){1,2})'
#             name_match = re.search(name_pattern, text)
            
#             name = name_match.group(1) if name_match else None
#             contact = email_match.group(0) if email_match else (phone_match.group(0) if phone_match else None)
            
#             # If no name found, try to use the first line of the text (for WhatsApp screenshots)
#             if not name:
#                 first_line = text.strip().splitlines()[0] if text.strip().splitlines() else None
#                 if first_line and len(first_line.split()) <= 4:  # Heuristic: likely a name if short
#                     name = first_line
            
#             logger.info(f"Extracted client info (regex/first line): {name}, {contact}")
#             return name, contact
            
#         except Exception as e:
#             logger.error(f"Error extracting client info: {str(e)}")
#             return None, None
    
#     def generate_response(self, text, is_followup=False):
#         """Generate AI response for the given text"""
#         if not text:
#             return {
#                 'content': "I couldn't understand your message. Please provide more information.",
#                 'from_knowledge_base': False,
#                 'knowledge_base_id': None
#             }
            
#         # For follow-up messages, skip knowledge base check
#         if not is_followup:
#             # Check knowledge base for existing response
#             kb_response = self.airtable_client.find_knowledge_base_response(text)
#             if kb_response:
#                 return {
#                     'content': kb_response['content'],
#                     'from_knowledge_base': True,
#                     'knowledge_base_id': kb_response['id']
#                 }
            
#         # If no knowledge base match or this is a follow-up, use OpenAI
#         try:
#             # Set appropriate system message based on whether this is a follow-up
#             system_message = "You are a helpful assistant for a business. Respond professionally and concisely to client inquiries. Provide accurate information and be helpful."
            
#             if is_followup:
#                 system_message = "You are a helpful assistant for a business. This is a follow-up question to a previous conversation. Respond professionally and concisely, addressing the specific follow-up question while maintaining context from the previous messages."
            
#             if USING_OPENAI_V1:
#                 response = self.client.chat.completions.create(
#                     model="gpt-4o",
#                     messages=[
#                         {"role": "system", "content": system_message},
#                         {"role": "user", "content": text}
#                     ],
#                     temperature=0.7,
#                     max_tokens=500
#                 )
#                 content = response.choices[0].message.content.strip()
#             else:
#                 response = openai.ChatCompletion.create(
#                     model="gpt-4o",
#                     messages=[
#                         {"role": "system", "content": system_message},
#                         {"role": "user", "content": text}
#                     ],
#                     temperature=0.7,
#                     max_tokens=500
#                 )
#                 content = response.choices[0].message.content.strip()
            
#             return {
#                 'content': content,
#                 'from_knowledge_base': False,
#                 'knowledge_base_id': None
#             }
#         except Exception as e:
#             logger.error(f"Error generating AI response: {str(e)}")
#             return {
#                 'content': f"I'm sorry, I encountered an error while processing your request. Please try again later.",
#                 'from_knowledge_base': False,
#                 'knowledge_base_id': None
#             }
    
#     def extract_text_from_image(self, image_path):
#         """Extract text from image using a hybrid pipeline.

#         Strategy:
#         1) Try EasyOCR with strong OpenCV preprocessing first (fast, robust for UIs like WhatsApp/Telegram/Email).
#         2) If the result is weak/empty, fall back to OpenAI Vision for semantic reading of tiny or blurry text.
#         3) If that still fails, use our aggressive grayscale/threshold fallback.
#         """
#         if not os.path.exists(image_path):
#             return "Error: Image file not found"
            
#         try:
#             # 1) Try EasyOCR path first
#             easy_text, easy_quality = self._extract_with_easyocr(image_path)
#             if easy_text and (len(easy_text) >= 25 or easy_quality >= 0.45):
#                 logger.info(f"EasyOCR succeeded (quality={easy_quality:.2f}, chars={len(easy_text)})")
#                 return easy_text

#             # 2) Optimize image before LMM processing
#             optimized_image = self._optimize_image(image_path)
            
#             # Start timing
#             start_time = time.time()
            
#             # Encode optimized image to base64
#             base64_image = base64.b64encode(optimized_image).decode('utf-8')
            
#             # Very detailed prompt for extracting text from unclear images
#             extraction_prompt = """
#             Extract ALL text from this image, even if the image is unclear, blurry, or has poor quality.
#             This is likely a screenshot from WhatsApp, Telegram, or another messaging app.
            
#             Please try very hard to read ANY text visible in the image, including:
#             - Message content
#             - Names and contact information
#             - Timestamps
#             - Any UI elements with text
            
#             If text is partially visible or unclear, make your best guess and include it.
#             Return ALL text content exactly as it appears, preserving the original formatting as much as possible.
#             """
            
#             # Set longer timeout for more thorough processing
#             timeout_seconds = 30  # 30 seconds timeout for better results
            
#             # Call OpenAI Vision API with optimized parameters
#             if USING_OPENAI_V1:
#                 response = self.client.chat.completions.create(
#                     model="gpt-4o",
#                     messages=[
#                         {
#                             "role": "system", 
#                             "content": "You are a specialized text extraction assistant focused on extracting text from unclear images. Your primary goal is to extract ANY visible text, even if it's difficult to read."
#                         },
#                         {
#                             "role": "user",
#                             "content": [
#                                 {"type": "text", "text": extraction_prompt},
#                                 {
#                                     "type": "image_url",
#                                     "image_url": {
#                                         "url": f"data:image/jpeg;base64,{base64_image}"
#                                     }
#                                 }
#                             ]
#                         }
#                     ],
#                     max_tokens=1000,
#                     timeout=timeout_seconds
#                 )
#                 extracted_text = response.choices[0].message.content.strip()
#             else:
#                 response = openai.ChatCompletion.create(
#                     model="gpt-4o",
#                     messages=[
#                         {
#                             "role": "system", 
#                             "content": "You are a specialized text extraction assistant focused on extracting text from unclear images. Your primary goal is to extract ANY visible text, even if it's difficult to read."
#                         },
#                         {
#                             "role": "user",
#                             "content": [
#                                 {"type": "text", "text": extraction_prompt},
#                                 {
#                                     "type": "image_url",
#                                     "image_url": {
#                                         "url": f"data:image/jpeg;base64,{base64_image}"
#                                     }
#                                 }
#                             ]
#                         }
#                     ],
#                     max_tokens=1000,
#                     request_timeout=timeout_seconds
#                 )
#                 extracted_text = response.choices[0].message.content.strip()
            
#             # Log processing time
#             processing_time = time.time() - start_time
#             logger.info(f"Image text extraction completed in {processing_time:.2f} seconds")
            
#             # If no text found or error message received, try fallback immediately
#             if (not extracted_text or 
#                 extracted_text.lower() in ["no text found.", "i don't see any text in this image."] or
#                 "sorry" in extracted_text.lower() or "can't" in extracted_text.lower()):
#                 logger.info("Primary extraction failed, trying fallback method")
#                 return self._extract_text_fallback(image_path)
                
#             return extracted_text
            
#         except Exception as e:
#             logger.error(f"Error extracting text from image with OpenAI: {str(e)}")
#             # Try fallback method
#             return self._extract_text_fallback(image_path)

#     def _preprocess_for_easyocr(self, image_path):
#         """Preprocess image for OCR using OpenCV. Returns numpy array suitable for EasyOCR."""
#         try:
#             if cv2 is None:
#                 # Fallback: simple PIL-based upscale and grayscale
#                 with Image.open(image_path) as img:
#                     if img.mode != 'RGB':
#                         img = img.convert('RGB')
#                     # Upscale for small text
#                     scale = 2 if max(img.size) < 1200 else 1
#                     if scale > 1:
#                         img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
#                     img = img.convert('L')
#                     return np.array(img)

#             # Use OpenCV for stronger preprocessing
#             img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
#             if img is None:
#                 # Fallback read
#                 with open(image_path, 'rb') as f:
#                     data = np.frombuffer(f.read(), np.uint8)
#                     img = cv2.imdecode(data, cv2.IMREAD_COLOR)

#             # Scale up small images
#             h, w = img.shape[:2]
#             if max(h, w) < 1200:
#                 scale = 1200.0 / max(h, w)
#                 img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

#             # Convert to grayscale
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#             # Denoise while keeping edges
#             gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

#             # Adaptive threshold to handle dark mode UIs
#             thresh = cv2.adaptiveThreshold(
#                 gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
#             )

#             # Morphological operations to strengthen characters
#             kernel = np.ones((2, 2), np.uint8)
#             processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

#             return processed
#         except Exception as e:
#             logger.error(f"Error in _preprocess_for_easyocr: {e}")
#             # Return simple grayscale as last resort
#             with Image.open(image_path) as img:
#                 return np.array(img.convert('L'))

#     def _extract_with_easyocr(self, image_path):
#         """Attempt text extraction with EasyOCR. Returns (text, quality_score)."""
#         if self._easyocr_reader is None:
#             return None, 0.0
#         try:
#             preprocessed = self._preprocess_for_easyocr(image_path)
#             result = self._easyocr_reader.readtext(preprocessed, detail=1, paragraph=True)
#             texts = []
#             confidences = []
#             for item in result:
#                 try:
#                     # item format: [bbox, text, confidence]
#                     texts.append(item[1])
#                     confidences.append(float(item[2]))
#                 except Exception:
#                     continue
#             combined_text = "\n".join(texts).strip()
#             avg_conf = float(np.mean(confidences)) if confidences else 0.0
#             logger.info(f"EasyOCR extracted {len(combined_text)} chars with avg_conf={avg_conf:.2f}")
#             return combined_text, avg_conf
#         except Exception as e:
#             logger.error(f"EasyOCR extraction error: {e}")
#             return None, 0.0
    
#     def _optimize_image(self, image_path):
#         """Optimize image for faster processing and better text extraction"""
#         try:
#             # Open the image
#             with Image.open(image_path) as img:
#                 # Apply image enhancement for better text clarity
#                 from PIL import ImageEnhance, ImageFilter
                
#                 # Convert to RGB if needed (handles RGBA, etc.)
#                 if img.mode != 'RGB':
#                     img = img.convert('RGB')
                
#                 # Apply a series of enhancements to improve text readability
#                 # Increase contrast
#                 enhancer = ImageEnhance.Contrast(img)
#                 img = enhancer.enhance(1.5)
                
#                 # Increase sharpness
#                 enhancer = ImageEnhance.Sharpness(img)
#                 img = enhancer.enhance(1.5)
                
#                 # Slightly increase brightness
#                 enhancer = ImageEnhance.Brightness(img)
#                 img = enhancer.enhance(1.2)
                
#                 # Apply a slight unsharp mask filter to enhance edges
#                 img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                
#                 # Resize large images to reduce processing time
#                 max_dimension = 1600  # Optimal size for text recognition
#                 if max(img.size) > max_dimension:
#                     # Calculate new dimensions while maintaining aspect ratio
#                     if img.width > img.height:
#                         new_width = max_dimension
#                         new_height = int(img.height * (max_dimension / img.width))
#                     else:
#                         new_height = max_dimension
#                         new_width = int(img.width * (max_dimension / img.height))
                    
#                     # Use LANCZOS for better quality resizing
#                     img = img.resize((new_width, new_height), Image.LANCZOS)
#                     logger.info(f"Resized image from {img.width}x{img.height} to {new_width}x{new_height}")
                
#                 # Save to bytes with high quality
#                 buffer = io.BytesIO()
#                 img.save(buffer, format='JPEG', quality=95, optimize=True)
#                 buffer.seek(0)
                
#                 return buffer.getvalue()
        
#         except Exception as e:
#             logger.error(f"Error optimizing image: {str(e)}")
#             # Return original image data if optimization fails
#             with open(image_path, "rb") as f:
#                 return f.read()
    
#     def _extract_text_fallback(self, image_path):
#         """Fallback method for text extraction using more aggressive approach"""
#         try:
#             # Try a different image enhancement approach
#             with Image.open(image_path) as img:
#                 # Convert to grayscale for better text contrast
#                 img = img.convert('L')
                
#                 # Apply threshold to make text more distinct
#                 from PIL import ImageOps
#                 img = ImageOps.autocontrast(img, cutoff=5)
                
#                 # Save to bytes
#                 buffer = io.BytesIO()
#                 img.save(buffer, format='PNG')
#                 buffer.seek(0)
#                 optimized_image = buffer.getvalue()
            
#             # Encode optimized image to base64
#             base64_image = base64.b64encode(optimized_image).decode('utf-8')
            
#             # Very aggressive fallback prompt
#             fallback_prompt = """
#             This is an image that may contain hard-to-read text. Please examine it EXTREMELY carefully.
            
#             I need you to extract ANY text visible in this image, even if:
#             - The image is blurry, low resolution, or poor quality
#             - The text is partially visible or cut off
#             - The text is small or in the background
            
#             Make your best attempt to read ANYTHING that looks like text.
#             If you see partial words or letters, include them and make your best guess.
#             This is likely a WhatsApp or messaging app screenshot.
#             """
            
#             # Set longer timeout for more thorough processing
#             timeout_seconds = 25  # 25 seconds timeout
            
#             # Call OpenAI Vision API with a different prompt and model configuration
#             if USING_OPENAI_V1:
#                 response = self.client.chat.completions.create(
#                     model="gpt-4o",
#                     messages=[
#                         {
#                             "role": "system", 
#                             "content": "You are a specialized OCR assistant with exceptional ability to read text from poor quality images. Your task is to extract ANY visible text, no matter how difficult it is to read."
#                         },
#                         {
#                             "role": "user",
#                             "content": [
#                                 {"type": "text", "text": fallback_prompt},
#                                 {
#                                     "type": "image_url",
#                                     "image_url": {
#                                         "url": f"data:image/jpeg;base64,{base64_image}"
#                                     }
#                                 }
#                             ]
#                         }
#                     ],
#                     max_tokens=1000,
#                     temperature=0.3,  # Lower temperature for more focused extraction
#                     timeout=timeout_seconds
#                 )
#                 description = response.choices[0].message.content.strip()
#             else:
#                 response = openai.ChatCompletion.create(
#                     model="gpt-4o",
#                     messages=[
#                         {
#                             "role": "system", 
#                             "content": "You are a specialized OCR assistant with exceptional ability to read text from poor quality images. Your task is to extract ANY visible text, no matter how difficult it is to read."
#                         },
#                         {
#                             "role": "user",
#                             "content": [
#                                 {"type": "text", "text": fallback_prompt},
#                                 {
#                                     "type": "image_url",
#                                     "image_url": {
#                                         "url": f"data:image/jpeg;base64,{base64_image}"
#                                     }
#                                 }
#                             ]
#                         }
#                     ],
#                     max_tokens=1000,
#                     temperature=0.3,  # Lower temperature for more focused extraction
#                     request_timeout=timeout_seconds
#                 )
#                 description = response.choices[0].message.content.strip()
            
#             # Always return model's best attempt, even if it includes caveats
#             if description and isinstance(description, str):
#                 return description.strip()
#             else:
#                 return ""
                
#         except Exception as e:
#             logger.error(f"Error in fallback text extraction: {str(e)}")
#             return f"Error extracting text from image: {str(e)}"

#     def summarize_image(self, image_path):
#         """Send the image directly to the model to transcribe as much as possible and provide a concise summary.

#         Returns a dict with keys: transcript, summary
#         """
#         if not os.path.exists(image_path):
#             return {"transcript": "", "summary": "Error: Image file not found"}

#         try:
#             optimized_image = self._optimize_image(image_path)
#             base64_image = base64.b64encode(optimized_image).decode('utf-8')

#             instruction = (
#                 "You are reading a screenshot (WhatsApp/Telegram/Email). First transcribe as much text as you can verbatim. "
#                 "Then provide a concise summary with key points and any action items. If some parts are unclear, make a best-effort reading and mark with [?]."
#             )

#             if USING_OPENAI_V1:
#                 response = self.client.chat.completions.create(
#                     model="gpt-4o",
#                     messages=[
#                         {"role": "system", "content": instruction},
#                         {
#                             "role": "user",
#                             "content": [
#                                 {"type": "text", "text": "Transcribe and summarize this screenshot."},
#                                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
#                             ],
#                         },
#                     ],
#                     max_tokens=1000,
#                     temperature=0.3,
#                 )
#                 content = response.choices[0].message.content.strip()
#             else:
#                 response = openai.ChatCompletion.create(
#                     model="gpt-4o",
#                     messages=[
#                         {"role": "system", "content": instruction},
#                         {
#                             "role": "user",
#                             "content": [
#                                 {"type": "text", "text": "Transcribe and summarize this screenshot."},
#                                 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
#                             ],
#                         },
#                     ],
#                     max_tokens=1000,
#                     temperature=0.3,
#                 )
#                 content = response.choices[0].message.content.strip()

#             # Heuristically split transcript and summary
#             transcript = content
#             summary = ""
#             # If the model separates with headings, try to parse
#             lower = content.lower()
#             if "summary:" in lower:
#                 parts = re.split(r"summary:\s*", content, flags=re.IGNORECASE)
#                 transcript = parts[0].strip()
#                 summary = parts[1].strip() if len(parts) > 1 else ""
#             elif "transcript:" in lower:
#                 parts = re.split(r"transcript:\s*", content, flags=re.IGNORECASE)
#                 transcript = parts[-1].strip()
#             return {"transcript": transcript, "summary": summary or transcript}
#         except Exception as e:
#             logger.error(f"summarize_image error: {e}")
#             return {"transcript": "", "summary": f"Error summarizing image: {e}"}

#     def approve_and_save(self, save_type, fields):
#         """
#         Save approved content to appropriate Airtable table
#         """
#         if not save_type or not fields:
#             return None, "Missing save type or fields"
            
#         result = None
#         try:
#             if save_type == 'Client_Log':
#                 result = self.airtable_client.create_record('Client_Log', fields)
#             elif save_type == 'Templates':
#                 result = self.airtable_client.create_record('Templates', fields)
#             elif save_type == 'Offers':
#                 result = self.airtable_client.create_record('Offers', fields)
#             else:
#                 return None, "Invalid save type"
                
#             if result:
#                 return result, f"Successfully saved to {save_type}"
#             else:
#                 return None, f"Failed to save to {save_type}"
                
#         except Exception as e:
#             logger.error(f"Error in approve_and_save: {str(e)}")
#             return None, f"Error: {str(e)}"

# class AirtableClient:
#     """Client for interacting with Airtable API"""
    
#     def __init__(self):
#         self.api_key = settings.AIRTABLE_API_KEY if hasattr(settings, 'AIRTABLE_API_KEY') else os.environ.get('AIRTABLE_API_KEY', '')
#         self.base_id = settings.AIRTABLE_BASE_ID if hasattr(settings, 'AIRTABLE_BASE_ID') else os.environ.get('AIRTABLE_BASE_ID', '')
#         self.headers = {
#             'Authorization': f'Bearer {self.api_key}',
#             'Content-Type': 'application/json'
#         }
#         self.base_url = f'https://api.airtable.com/v0/{self.base_id}'
    
#     def check_client_exists(self, contact_info, message_content=None):
#         """Check if client exists in Clients or Leads tables with enhanced search"""
#         if not contact_info and not message_content:
#             return None, None
        
#         # Try to find in Clients table - more comprehensive search
#         if contact_info:
#             client_filters = self._build_search_filters(contact_info)
#             for filter_query in client_filters:
#                 clients = self.get_table_records('Clients', filter_query)
#                 if clients:
#                     logger.info(f"Found client in Clients table: {clients[0].get('id')}")
#                     return clients[0], 'Clients'
        
#         # Try to find in Leads table - more comprehensive search
#         if contact_info:
#             client_filters = self._build_search_filters(contact_info)
#             for filter_query in client_filters:
#                 leads = self.get_table_records('Leads', filter_query)
#                 if leads:
#                     logger.info(f"Found client in Leads table: {leads[0].get('id')}")
#                     return leads[0], 'Leads'
        
#         # If message content is provided, also check for duplicate content
#         if message_content:
#             # Check for duplicate content in Leads table
#             all_leads = self.get_table_records('Leads')
#             for lead in all_leads:
#                 fields = lead.get('fields', {})
#                 lead_message = fields.get('Message', '') or fields.get('Notes', '')
#                 if lead_message and self._is_duplicate_content(message_content, lead_message):
#                     logger.info(f"Found duplicate content in Leads table: {lead.get('id')}")
#                     # Return a special flag for image/text duplicate
#                     lead['is_message_duplicate'] = True
#                     return lead, 'Leads'
        
#         logger.info(f"No client found for contact info: {contact_info}")
#         return None, None


#     # def generate_image_hash(image_file):
#     #     hasher = hashlib.sha256()
#     #     for chunk in image_file.chunks():
#     #         hasher.update(chunk)
#     #     return hasher.hexdigest()


#     def _normalize_text(self, text):
#         if not text:
#             return ''
#         t = text.lower()
#         # Remove boilerplate extraction phrases
#         t = re.sub(r"sure,? here'?s the (extracted )?text .*?image:?", ' ', t)
#         t = re.sub(r"\btranscription:?\b", ' ', t)
#         t = re.sub(r"\bmessages?:?\b", ' ', t)
#         t = re.sub(r"\bheader:?\b", ' ', t)
#         t = re.sub(r"from the image:?", ' ', t)
#         # Remove timestamps and clock times
#         t = re.sub(r"\b\d{1,2}:\d{2}\s*(am|pm)?\b", ' ', t)
#         # Remove extra bullets/markdown
#         t = t.replace('*', ' ').replace('`', ' ').replace('—', ' ').replace('–', ' ')
#         # Collapse non-alphanumeric except @ and + (for emails/phones) to spaces
#         t = re.sub(r"[^a-z0-9@+\s]", ' ', t)
#         # Collapse whitespace
#         t = re.sub(r"\s+", ' ', t).strip()
#         return t

#     def check_lead_message_duplicate(self, message_content):
#         norm_new = self._normalize_text(message_content)
#         if not norm_new:
#             return False
#         all_leads = self.get_table_records('Leads')
#         for lead in all_leads:
#             fields = lead.get('fields', {})
#             lead_message = fields.get('Message', '') or fields.get('Notes', '')
#             norm_existing = self._normalize_text(lead_message)
#             if not norm_existing:
#                 continue
#             # Use fuzzy comparison for robustness
#             if self._is_duplicate_content(norm_new, norm_existing, threshold=0.7):
#                 return True
#         return False
    
#     def _is_duplicate_content(self, content1, content2, threshold=0.8):
#         """Check if two content strings are duplicates based on similarity"""
#         if not content1 or not content2:
#             return False
        
#         # Clean and normalize content
#         content1_clean = content1.lower().strip()
#         content2_clean = content2.lower().strip()
        
#         # Exact match
#         if content1_clean == content2_clean:
#             return True
        
#         # Check if one is contained in the other (for partial duplicates)
#         if content1_clean in content2_clean or content2_clean in content1_clean:
#             return True
        
#         # Simple word overlap check
#         words1 = set(content1_clean.split())
#         words2 = set(content2_clean.split())
        
#         if len(words1) > 0 and len(words2) > 0:
#             common_words = words1.intersection(words2)
#             similarity = len(common_words) / max(len(words1), len(words2))
#             return similarity >= threshold
        
#         return False
    
#     def create_lead(self, name, contact_info, message=None):
#         """Create a new lead in the Leads table"""
#         if not name or not contact_info:
#             return None
            
#         fields = {
#             'Name': name,
#         }
        
#         # Determine if contact_info is email or phone
#         if '@' in contact_info:
#             fields['Email'] = contact_info
#         else:
#             fields['Phone'] = contact_info
            
#         if message:
#             fields['Notes'] = message
            
#         return self.create_record('Leads', fields)
    
#     def log_message(self, client_id, client_table, message_content, response_content=None, source='Manual'):
#         """Log a message in the Client_Log table"""
#         if not client_id or not message_content:
#             return None
            
#         fields = {
#             'Client': [client_id],  # Airtable linked record format
#             'Message': message_content,
#             'Source': source,
#             'Type': 'Message'
#         }
        
#         if response_content:
#             fields['Response'] = self._format_response_for_airtable(response_content)
            
#         return self.create_record('Client_Log', fields)
    
#     def save_to_knowledge_base(self, client_message, approved_reply):
#         """Save a message and its approved reply to the Knowledge_Base table"""
#         if not client_message or not approved_reply:
#             return None
            
#         fields = {
#             'Client_Message': client_message,
#             'Final_Approved_Reply': self._format_response_for_airtable(approved_reply)
#         }
        
#         return self.create_record('Knowledge_Base', fields)
    
#     def find_knowledge_base_response(self, query):
#         """Search Knowledge_Base table for matching response"""
#         if not query or query.strip() == "":
#             return None
            
#         # Get all knowledge base entries
#         knowledge_entries = self.get_table_records('Knowledge_Base')
#         if not knowledge_entries:
#             return None
            
#         best_match = None
#         best_score = 0
        
#         # Simple text matching algorithm
#         query_lower = query.lower()
#         query_words = set(query_lower.split())
        
#         for entry in knowledge_entries:
#             fields = entry.get('fields', {})
#             question = fields.get('Client_Message', '')
#             answer = fields.get('Final_Approved_Reply', '')
            
#             if not question or not answer:
#                 continue
                
#             # Direct match - return immediately
#             if query_lower == question.lower():
#                 return {
#                     'content': answer,
#                     'id': entry.get('id')
#                 }
                
#             # Word overlap scoring
#             question_words = set(question.lower().split())
#             common_words = query_words.intersection(question_words)
            
#             # Calculate score based on word overlap percentage
#             if len(query_words) > 0 and len(question_words) > 0:
#                 overlap_score = len(common_words) / max(len(query_words), len(question_words))
                
#                 # Check for substantial keyword match
#                 if overlap_score > best_score:
#                     best_score = overlap_score
#                     best_match = {
#                         'content': answer,
#                         'id': entry.get('id')
#                     }
        
#         # Return best match if it exceeds threshold
#         if best_score > 0.6:  # At least 60% word overlap
#             return best_match
                
#         return None
    
#     def get_table_records(self, table_name, filters=None):
#         """Get records from a specific table with optional filters"""
#         url = f"{self.base_url}/{table_name}"
        
#         params = {}
#         if filters:
#             filter_formula = self._build_filter_formula(filters)
#             params['filterByFormula'] = filter_formula
            
#         try:
#             response = requests.get(url, headers=self.headers, params=params)
#             if response.status_code == 200:
#                 return response.json().get('records', [])
#             else:
#                 logger.error(f"Airtable API Error: {response.status_code}")
#                 return []
#         except Exception as e:
#             logger.error(f"Error getting records from Airtable: {str(e)}")
#             return []
    
#     def _build_filter_formula(self, filters):
#         """Build Airtable filter formula from filters dict"""
#         conditions = []
#         for field, value in filters.items():
#             conditions.append(f"{{{field}}} = '{value}'")
        
#         if len(conditions) == 1:
#             return conditions[0]
#         else:
#             return f"AND({','.join(conditions)})"
    
#     def create_record(self, table_name, fields):
#         """Create a new record in the specified table"""
#         url = f"{self.base_url}/{table_name}"
#         data = {"fields": fields}
        
#         try:
#             response = requests.post(url, headers=self.headers, json=data)
#             if response.status_code in [200, 201]:
#                 return response.json()
#             else:
#                 # Log full error payload for diagnosis
#                 err_text = None
#                 try:
#                     err_text = response.text
#                 except Exception:
#                     pass
#                 logger.error(f"Error creating record [{table_name}] {response.status_code}: {err_text}")
#                 return None
#         except Exception as e:
#             logger.error(f"Exception creating record: {str(e)}")
#             return None

#     def create_record_with_fallback(self, table_name, fields):
#         """Attempt to create a record; if it fails (e.g., unknown field names or single-select issues),
#         retry with common alternative field names.
#         """
#         # First try as-is
#         result = self.create_record(table_name, fields)
#         if result:
#             return result

#         # Build variants with alternate field names
#         key_variants = {
#             'Full_Name': ['Full_Name', 'Full Name', 'Name'],
#             'Phone': ['Phone', 'Phone Number', 'Mobile', 'Contact', 'Contact_Number'],
#             'Email': ['Email', 'E-mail', 'Email Address'],
#             'Message': ['Message', 'Notes', 'Description'],
#             'Source': ['Source', 'Channel', 'Platform'],
#             'Status': ['Status', 'Lead Status', 'Stage'],
#         }

#         # Generate a small set of reasonable permutations
#         from itertools import product
#         base = {
#             'Full_Name': fields.get('Full_Name'),
#             'Phone': fields.get('Phone'),
#             'Email': fields.get('Email'),
#             'Message': fields.get('Message'),
#             'Source': fields.get('Source'),
#             'Status': fields.get('Status'),
#         }

#         variants_lists = []
#         for key, options in key_variants.items():
#             value = base.get(key)
#             if value is None:
#                 variants_lists.append([None])
#             else:
#                 variants_lists.append([(opt, value) for opt in options])

#         # Create combinations only for keys that have values
#         combos = []
#         for combo in product(*variants_lists):
#             variant = {}
#             for item in combo:
#                 if item and item[1] is not None:
#                     variant[item[0]] = item[1]
#             if variant:
#                 combos.append(variant)

#         # Ensure original order first, then variants
#         tried = set()
#         for variant_fields in [fields] + combos[:12]:  # cap to avoid too many attempts
#             sig = tuple(sorted(variant_fields.keys()))
#             if sig in tried:
#                 continue
#             tried.add(sig)
#             result = self.create_record(table_name, variant_fields)
#             if result:
#                 return result

#         return None

#     def _build_search_filters(self, search_term):
#         """Build multiple search filters to increase chance of finding a match"""
#         filters = []
        
#         # Clean up the search term
#         search_term = search_term.strip()
        
#         # Direct matches
#         if '@' in search_term:  # Looks like an email
#             filters.append({'Email': search_term})
        
#         # Phone number search (try different formats)
#         phone_clean = ''.join(filter(str.isdigit, search_term))
#         if len(phone_clean) >= 7:  # Minimum length for a phone number
#             filters.append({'Phone': search_term})
#             # Try with just the digits
#             if phone_clean != search_term:
#                 filters.append({'Phone': phone_clean})
        
#         # Name search
#         if len(search_term.split()) >= 2:  # Might be a full name
#             filters.append({'Name': search_term})
        
#         # Add a general search filter for partial matches
#         if len(search_term) > 3:  # Avoid very short terms
#             # Note: This is a simplified approach - Airtable's actual search capabilities may vary
#             filters.append({'Name': f"FIND('{search_term}')"})
        
#         return filters

#     def get_client_details(self, client_id, table_name):
#         """Get detailed client information including history"""
#         if not client_id or not table_name:
#             return None
        
#         # Get client record
#         client = None
#         try:
#             url = f"{self.base_url}/{table_name}/{client_id}"
#             response = requests.get(url, headers=self.headers)
#             if response.status_code == 200:
#                 client = response.json()
#             else:
#                 logger.error(f"Error getting client details: {response.status_code}")
#                 return None
#         except Exception as e:
#             logger.error(f"Exception getting client details: {str(e)}")
#             return None
        
#         if not client:
#             return None
        
#         # Get client history from Client_Log
#         client_history = []
#         try:
#             # Use FILTERBYFORMULA to find logs for this client
#             formula = f"Client='{client_id}'"
#             url = f"{self.base_url}/Client_Log?filterByFormula={formula}"
#             response = requests.get(url, headers=self.headers)
#             if response.status_code == 200:
#                 logs = response.json().get('records', [])
#                 client_history = sorted(logs, key=lambda x: x.get('createdTime', ''), reverse=True)
#             else:
#                 logger.error(f"Error getting client history: {response.status_code}")
#         except Exception as e:
#             logger.error(f"Exception getting client history: {str(e)}")
        
#         # Combine client info with history
#         result = {
#             'client': client,
#             'history': client_history,
#             'table': table_name
#         }
        
#         return result 

#     def search_knowledge_base(self, query, threshold=0.6):
#         """Enhanced search in Knowledge_Base table for matching responses"""
#         if not query or query.strip() == "":
#             return []
            
#         # Get all knowledge base entries
#         knowledge_entries = self.get_table_records('Knowledge_Base')
#         if not knowledge_entries:
#             return []
            
#         matches = []
        
#         # Enhanced text matching algorithm
#         query_lower = query.lower()
#         query_words = set(query_lower.split())
        
#         for entry in knowledge_entries:
#             fields = entry.get('fields', {})
#             question = fields.get('Client_Message', '')
#             answer = fields.get('Final_Approved_Reply', '')
#             category = fields.get('Category', '')
            
#             if not question or not answer:
#                 continue
                
#             # Direct match - highest priority
#             if query_lower == question.lower():
#                 matches.append({
#                     'content': answer,
#                     'id': entry.get('id'),
#                     'question': question,
#                     'score': 1.0,
#                     'category': category
#                 })
#                 continue
                
#             # Partial match - check if query is contained in question
#             if query_lower in question.lower():
#                 matches.append({
#                     'content': answer,
#                     'id': entry.get('id'),
#                     'question': question,
#                     'score': 0.9,
#                     'category': category
#                 })
#                 continue
                
#             # Word overlap scoring
#             question_words = set(question.lower().split())
#             common_words = query_words.intersection(question_words)
            
#             # Calculate score based on word overlap percentage
#             if len(query_words) > 0 and len(question_words) > 0:
#                 overlap_score = len(common_words) / max(len(query_words), len(question_words))
                
#                 # Check if score exceeds threshold
#                 if overlap_score > threshold:
#                     matches.append({
#                         'content': answer,
#                         'id': entry.get('id'),
#                         'question': question,
#                         'score': overlap_score,
#                         'category': category
#                     })
        
#         # Sort matches by score (highest first)
#         matches.sort(key=lambda x: x['score'], reverse=True)
        
#         return matches 

#     def search_accounts(self, query, threshold=0.6):
#         """Search Accounts table for records matching the query in any field."""
#         if not query or query.strip() == "":
#             return []
        
#         # Get all account entries
#         account_entries = self.get_table_records('Accounts')
#         if not account_entries:
#             return []
        
#         matches = []
#         query_lower = query.lower()
#         query_words = set(query_lower.split())
        
#         for entry in account_entries:
#             fields = entry.get('fields', {})
#             # Combine all field values into a single string for matching
#             combined = ' '.join(str(v) for v in fields.values() if v)
#             combined_lower = combined.lower()
            
#             # Direct match
#             if query_lower == combined_lower:
#                 matches.append({
#                     'fields': fields,
#                     'id': entry.get('id'),
#                     'score': 1.0
#                 })
#                 continue
#             # Partial match
#             if query_lower in combined_lower:
#                 matches.append({
#                     'fields': fields,
#                     'id': entry.get('id'),
#                     'score': 0.9
#                 })
#                 continue
#             # Word overlap
#             combined_words = set(combined_lower.split())
#             common_words = query_words.intersection(combined_words)
#             if len(query_words) > 0 and len(combined_words) > 0:
#                 overlap_score = len(common_words) / max(len(query_words), len(combined_words))
#                 if overlap_score > threshold:
#                     matches.append({
#                         'fields': fields,
#                         'id': entry.get('id'),
#                         'score': overlap_score
#                     })
#         matches.sort(key=lambda x: x['score'], reverse=True)
#         return matches

#     def _format_response_for_airtable(self, response):
#         """Format response text for Airtable in a professional email-like format"""
#         if not response:
#             return ""
            
#         # Check if the response already has a subject line
#         if response.startswith("Subject:"):
#             return response
            
#         # Format the response with a standard subject line and signature
#         formatted_response = "Subject: Assistance with Your Recent Inquiry\n\n"
        
#         # Add salutation if not present
#         if not any(greeting in response[:50].lower() for greeting in ["dear", "hello", "hi ", "greetings"]):
#             formatted_response += "Dear Client,\n\n"
            
#         # Add the main content
#         formatted_response += response.strip()
        
#         # Add signature if not present
#         if not any(closing in response[-100:].lower() for closing in ["regards", "sincerely", "thank you", "best", "warm"]):
#             formatted_response += "\n\nWarm regards,\n\n[Your Name]\n[Your Position]\n[Your Contact Information]"
            
#         return formatted_response 

# def generate_image_hash(image_file):
#         hasher = hashlib.sha256()
#         for chunk in image_file.chunks():
#             hasher.update(chunk)
#         return hasher.hexdigest()
