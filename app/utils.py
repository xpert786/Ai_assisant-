import os
import json
from PIL import Image
import re
from django.conf import settings
import logging
import base64
import openai
import io
import requests
import time
import hashlib
import numpy as np
from importlib.metadata import version, PackageNotFoundError

# Optional heavy deps
try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import easyocr  # type: ignore
except Exception:  # pragma: no cover
    easyocr = None

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ai_assistant.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Check OpenAI API availability with proper error handling
try:
    openai_version = version("openai")
    # Check if v1 API methods are available
    USING_OPENAI_V1 = hasattr(openai, 'OpenAI')
    logger.info(f"Detected OpenAI version: {openai_version}, Using V1 API: {USING_OPENAI_V1}")
except PackageNotFoundError:
    logger.warning("OpenAI package not found, defaulting to V0 API format")
    openai_version = 'unknown'
    USING_OPENAI_V1 = False  # Default to V0 for compatibility
except Exception as e:
    logger.error(f"Error detecting OpenAI version: {e}, defaulting to V0 API")
    openai_version = 'unknown'
    USING_OPENAI_V1 = False  # Default to V0 for compatibility

# OpenAI configuration

class AIClient:
    """Client for interacting with OpenAI API"""
    
    def __init__(self):
        self.api_key = settings.OPENAI_API_KEY if hasattr(settings, 'OPENAI_API_KEY') else os.environ.get('OPENAI_API_KEY', '')
        # Use fastest models available for maximum speed
        self.text_model = getattr(settings, 'OPENAI_MODEL', 'gpt-3.5-turbo') or 'gpt-3.5-turbo'
        self.vision_model = getattr(settings, 'OPENAI_VISION_MODEL', 'gpt-4-vision-preview') or 'gpt-4-vision-preview'
        self.model = self.vision_model

        # Initialize OpenAI client with fallback logic
        if not self.api_key:
            logger.error("OpenAI API key not found. Please set OPENAI_API_KEY in settings or environment.")
            raise ValueError("OpenAI API key is required")

        # Try to initialize v1 client first
        self.client = None
        try:
            self.client = openai.OpenAI(api_key=self.api_key)
            self.using_v1 = True
            logger.info(f"AIClient initialized with V1 API - models: text={self.text_model}, vision={self.vision_model}")
        except Exception as e:
            # Fall back to v0 API
            openai.api_key = self.api_key
            self.using_v1 = False
            logger.info(f"V1 API not available ({e}), using V0 API - models: text={self.text_model}, vision={self.vision_model}")
        self.airtable_client = AirtableClient()
        self.sender_name = getattr(settings, 'RESPONSE_SENDER_NAME', 'Your Name')

        # Initialize OpenAI client
        self._easyocr_reader = None
        
    def extract_client_info(self, text):
        """Extract client name and contact info from text"""
        if not text:
            logger.debug("No text provided for client info extraction")
            return None, None

        try:
            logger.info(f"Extracting client info from text: {text[:100]}...")

            # Use appropriate OpenAI API based on version
            if self.using_v1:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that extracts client information from text. Extract the name and contact information (email or phone) if present. Return in JSON format with keys 'name' and 'contact'."},
                        {"role": "user", "content": text}
                    ],
                    temperature=0.3,
                    max_tokens=150
                )
                content = response.choices[0].message.content.strip()
            else:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that extracts client information from text. Extract the name and contact information (email or phone) if present. Return in JSON format with keys 'name' and 'contact'."},
                        {"role": "user", "content": text}
                    ],
                    temperature=0.3,
                    max_tokens=150
                )
                content = response.choices[0].message.content.strip()


            logger.debug(f"OpenAI response for client extraction: {content}")
            
            # Try to parse JSON from the response
            try:
                data = json.loads(content)
                name = data.get('name')
                contact = data.get('contact')
                
                # Validate extracted information
                if name and len(name) > 2 and contact and len(contact) > 5:
                    logger.info(f"Extracted client info: {name}, {contact}")
                    return name, contact
            except json.JSONDecodeError:
                # If not valid JSON, try regex extraction
                pass
                
            # Fallback to regex extraction
            # Look for email
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            email_match = re.search(email_pattern, text)
            
            # Look for phone number (simple pattern)
            phone_pattern = r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b'
            phone_match = re.search(phone_pattern, text)
            

            # Enhanced patterns to catch various name formats
            name_patterns = [
                # WhatsApp group name with company and people
                r'([A-Z][A-Za-z0-9\s&\-\.]+(?:LLC|Ltd|Corporation|Corp|Inc|Limited|Trading|Solutions|Group))(?:\s*[-:]\s*([A-Z][A-Za-z\s&]+))?',
                
                # Direct message names or mentions
                r'(?:^|\n)([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)(?=:|\n|$)',
                
                # Self-introduction patterns
                r'(?:I am|my name is|this is|Hi,?\s+I\'?m|Hello,?\s+I\'?m)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)',
                
                # Email signature style names
                r'(?:Regards|Best|Sincerely|Thanks|Thank you|Yours),?\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)',
                
                # Company name with contact person
                r'(?:contact|attention|attn|c/o|care of):\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)',
                
                # Name with title
                r'(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)',
                
                # Group chat member list
                r'Members:\s*(?:[0-9]+\s*)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)'
            ]
            
            name = None
            for pattern in name_patterns:
                name_match = re.search(pattern, text)
                if name_match:
                    if name_match.group(2):  # If there's a second group (person name after company)
                        name = name_match.group(2)
                    else:
                        name = name_match.group(1)
                    break
            contact = email_match.group(0) if email_match else (phone_match.group(0) if phone_match else None)
            
            # If no name found, try to use the first line of the text (for WhatsApp screenshots)
            if not name:
                first_line = text.strip().splitlines()[0] if text.strip().splitlines() else None
                if first_line and len(first_line.split()) <= 4:  # Heuristic: likely a name if short
                    name = first_line
            
            logger.info(f"Extracted client info (regex/first line): {name}, {contact}")
            return name, contact


        except Exception as e:
            logger.error(f"Error extracting client info: {e}")
            return None, None

    def detect_source_from_content(self, content):
        """Detect the likely source (WhatsApp, Email, Telegram) from message content."""
        if not content:
            return None

        content_lower = content.lower()

        # WhatsApp indicators
        if any(x in content_lower for x in ["whatsapp", "last seen", "typing…", "online", "wa.me/"]):
            return "WhatsApp"

        # Telegram indicators
        if any(x in content_lower for x in ["telegram", "t.me/", "forwarded message", "telegram channel"]):
            return "Telegram"

        # Email indicators
        if any(x in content_lower for x in ["subject:", "from:", "to:", "cc:", "bcc:", "forwarded message", "@gmail.com", "@outlook.com", "@yahoo.com"]):
            return "Email"

        # Default - don't set any source if we can't detect it
        return None

    def detect_language_openai(self, text):
        """Detect the language of the given text using OpenAI"""
        if not text:
            return 'English'  # Default to English
        
        try:
            # Use appropriate OpenAI API based on version
            if self.using_v1:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a language detection assistant. Given a text, determine the primary language and return only the language name in English (e.g., 'English', 'Spanish', 'French', 'Chinese', 'Arabic', etc.). Keep your response to just the language name."},
                        {"role": "user", "content": f"Detect the language of this text: {text[:500]}"}
                    ],
                    temperature=0.1,
                    max_tokens=50
                )
                language = response.choices[0].message.content.strip()
            else:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a language detection assistant. Given a text, determine the primary language and return only the language name in English (e.g., 'English', 'Spanish', 'French', 'Chinese', 'Arabic', etc.). Keep your response to just the language name."},
                        {"role": "user", "content": f"Detect the language of this text: {text[:500]}"}
                    ],
                    temperature=0.1,
                    max_tokens=50
                )
                language = response.choices[0].message.content.strip()
            
            # Clean up the response to ensure it's a valid language name
            language = language.replace('.', '').replace('"', '').replace("'", '').strip()
            
            # Map common variations to standard names
            language_mapping = {
                'chinese': 'Chinese',
                'mandarin': 'Chinese',
                'cantonese': 'Chinese',
                'spanish': 'Spanish',
                'french': 'French',
                'german': 'German',
                'italian': 'Italian',
                'portuguese': 'Portuguese',
                'russian': 'Russian',
                'japanese': 'Japanese',
                'korean': 'Korean',
                'arabic': 'Arabic',
                'hindi': 'Hindi',
                'english': 'English'
            }
            
            detected_language = language_mapping.get(language.lower(), language.title())
            logger.info(f"Detected language: {detected_language}")
            return detected_language
            
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return 'English'  # Default fallback

    def generate_response(self, text, is_followup=False, recipient_name: str | None = None, conversation_history=None, detected_language='English'):
        """Generate AI response for the given text in the detected language"""
        if not text:
            # Return error message in detected language
            error_messages = {
                'Italian': "Non sono riuscito a capire il tuo messaggio. Per favore fornisci più informazioni.",
                'Spanish': "No pude entender tu mensaje. Por favor proporciona más información.",
                'Portuguese': "Não consegui entender sua mensagem. Por favor, forneça mais informações.",
                'English': "I couldn't understand your message. Please provide more information."
            }
            return {
                'content': error_messages.get(detected_language, error_messages['English']),
                'from_knowledge_base': False,
                'knowledge_base_id': None
            }
            
        # For follow-up messages, skip knowledge base check
        if not is_followup:
            # Check knowledge base for existing response
            kb_response = self.airtable_client.find_knowledge_base_response(text)
            if kb_response:
                return {
                    'content': kb_response['content'],
                    'from_knowledge_base': True,
                    'knowledge_base_id': kb_response['id']
                }
            
        # If no knowledge base match or this is a follow-up, use OpenAI
        try:
            # Build conversation context
            messages = []
            
            # Set appropriate system message based on whether this is a follow-up and detected language
            language_instructions = {
                'Italian': "Rispondi in italiano in modo naturale e conversazionale, come ChatGPT. Sii amichevole, utile e coinvolgente. Non usare firme formali o saluti come 'Cordiali saluti' o 'Ciao'. Rispondi semplicemente in modo naturale al messaggio dell'utente.",
                'Spanish': "Responde en español de manera natural y conversacional, como ChatGPT. Sé amigable, útil y atractivo. No uses firmas formales o saludos como 'Saludos cordiales' o 'Hola'. Simplemente responde de manera natural al mensaje del usuario.",
                'Portuguese': "Responda em português de forma natural e conversacional, como ChatGPT. Seja amigável, útil e envolvente. Não use assinaturas formais ou cumprimentos como 'Atenciosamente' ou 'Olá'. Simplesmente responda naturalmente à mensagem do usuário.",
                'English': "You are a helpful AI assistant. Respond naturally and conversationally, like ChatGPT. Be friendly, helpful, and engaging. Don't use formal signatures or greetings like 'Best regards' or 'Hello'. Just respond naturally to the user's message."
            }
            
            system_message = language_instructions.get(detected_language, language_instructions['English'])
            
            if is_followup:
                followup_instructions = {
                    'Italian': "Questo è un follow-up di una conversazione precedente. Rispondi in italiano in modo naturale e conversazionale, come ChatGPT. Sii amichevole, utile e coinvolgente. Non usare firme formali o saluti. Rispondi semplicemente in modo naturale al messaggio dell'utente mantenendo il contesto della conversazione precedente.",
                    'Spanish': "Esta es una pregunta de seguimiento de una conversación anterior. Responde en español de manera natural y conversacional, como ChatGPT. Sé amigable, útil y atractivo. No uses firmas formales o saludos. Simplemente responde de manera natural al mensaje del usuario manteniendo el contexto de la conversación anterior.",
                    'Portuguese': "Esta é uma pergunta de acompanhamento de uma conversa anterior. Responda em português de forma natural e conversacional, como ChatGPT. Seja amigável, útil e envolvente. Não use assinaturas formais ou cumprimentos. Simplesmente responda naturalmente à mensagem do usuário mantendo o contexto da conversa anterior.",
                    'English': "You are a helpful AI assistant. This is a follow-up question to a previous conversation. Respond naturally and conversationally, like ChatGPT. Be friendly, helpful, and engaging. Don't use formal signatures or greetings. Just respond naturally to the user's message while maintaining context from the previous conversation."
                }
                system_message = followup_instructions.get(detected_language, followup_instructions['English'])
            
            messages.append({"role": "system", "content": system_message})
            
            # Add conversation history if provided
            if conversation_history:
                for msg in conversation_history:
                    if msg.get('role') and msg.get('content'):
                        messages.append({
                            "role": msg['role'],
                            "content": msg['content']
                        })
            
            # Add current user message
            messages.append({"role": "user", "content": text})
            

            logger.info(f"Generating AI response for text: {text[:100]}...")
            if self.using_v1:
                response = self.client.chat.completions.create(
                    model=self.text_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=800
                )
                content = response.choices[0].message.content.strip()
            else:
                response = openai.ChatCompletion.create(
                    model=self.text_model,
                    messages=messages,

                    temperature=0.7,
                    max_tokens=800
                )
                content = response.choices[0].message.content.strip()
            logger.debug(f"AI response generated: {content[:100]}...")
            
            # For real-time chat, don't format with professional reply
            if is_followup and conversation_history:
                return {
                    'content': content,
                    'from_knowledge_base': False,
                    'knowledge_base_id': None
                }
            else:
                formatted = self._format_professional_reply(content, recipient_name)
                return {
                    'content': formatted,
                    'from_knowledge_base': False,
                    'knowledge_base_id': None
                }
        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            return {
                'content': f"I'm sorry, I encountered an error while processing your request. Please try again later.",
                'from_knowledge_base': False,
                'knowledge_base_id': None
            }
    
    def extract_text_from_image(self, image_path):
        """Extract text from image using a streamlined pipeline for faster processing.
        Skips EasyOCR and goes directly to Vision API for better speed.
        """
        if not os.path.exists(image_path):
            return "Error: Image file not found"
            
        try:
            # Skip EasyOCR and go directly to Vision API for faster processing
            # Optimize image before LMM processing
            optimized_image = self._optimize_image(image_path)
            
            # Start timing
            start_time = time.time()
            
            # Encode optimized image to base64
            base64_image = base64.b64encode(optimized_image).decode('utf-8')
            
            # Ultra-simplified prompt for maximum speed - also instruct NOT to include preamble like "Sure, here is the text..."
            extraction_prompt = """Extract all text from this image. Focus on names, contact info, and message content. 
            IMPORTANT: Do NOT include phrases like "Sure, here is the text extracted from the image:" or any similar preamble. 
            Just provide the raw extracted text exactly as it appears in the image."""
            
            # Minimal timeout for maximum speed
            timeout_seconds = 15
            
            # Call OpenAI Vision API with optimized parameters

            logger.info(f"Extracting text from image using vision API (timeout: {timeout_seconds}s)")
            if self.using_v1:
                response = self.client.chat.completions.create(
                    model=self.vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": extraction_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=300,
                    request_timeout=timeout_seconds
                )
                extracted_text = response.choices[0].message.content.strip()
            else:
                response = openai.ChatCompletion.create(
                    model=self.vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": extraction_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=300,
                    request_timeout=timeout_seconds
                )
                extracted_text = response.choices[0].message.content.strip()

            logger.debug(f"Text extracted from image: {extracted_text[:100]}...")
            
            # Log processing time
            processing_time = time.time() - start_time
            logger.info(f"Image text extraction completed in {processing_time:.2f} seconds")
            
            # If no text found or error message received, try fallback immediately
            if (not extracted_text or 
                extracted_text.lower() in ["no text found.", "i don't see any text in this image."] or
                "sorry" in extracted_text.lower() or "can't" in extracted_text.lower()):
                logger.info("Primary extraction failed, trying fallback method")
                # Try tiled EasyOCR first for long/large chats
                tiled_text, tiled_conf = self._extract_with_easyocr_tiled(image_path)
                tiled_text = self._clean_extracted_text(tiled_text or "")
                if tiled_text and (len(tiled_text) > 25 or (tiled_conf or 0.0) >= 0.3):
                    return tiled_text
                return self._clean_extracted_text(self._extract_text_fallback(image_path))
                
            vision_text = self._clean_extracted_text(extracted_text)
            vision_score = 0.6 + (len(vision_text) / 900.0)

            best_score = 0.0  # Initialize best_score
            best_text = ""    # Initialize best_text
            if vision_score >= best_score:
                return vision_text
            return best_text or vision_text
            
        except Exception as e:
            logger.error(f"Error extracting text from image with OpenAI: {str(e)}")
            # Try fallback method
            return self._clean_extracted_text(self._extract_text_fallback(image_path))

    def _clean_extracted_text(self, text: str) -> str:
        """Clean model/EasyOCR output: remove headings like 'Extracted Text:' or 'Transcription:' etc.
        Keeps only the raw message content.
        """
        if not text:
            return text
        try:
            cleaned = text.strip()
            # Remove common lead-in phrases produced by assistants
            cleaned = re.sub(r"^(?:sure,?\s*)?here (?:is|are) (?:the )?(?:extracted|transcribed|transcription of the) text[^:]*:\s*",
                             "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"^(?:transcript|transcription|extracted text)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
            # Remove generic failure statements
            if re.search(r"unable to extract|don't see any text|no text found|cannot read the text", cleaned, re.IGNORECASE):
                cleaned = ""
            # Remove markdown decorations
            cleaned = cleaned.replace('```', ' ').replace('**', ' ').strip()
            return cleaned
        except Exception:
            return text

    def _preprocess_for_easyocr(self, image_path):
        """Preprocess image for OCR using OpenCV. Returns numpy array suitable for EasyOCR."""
        try:
            if cv2 is None:
                # Fallback: simple PIL-based upscale and grayscale
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Upscale for small text
                    scale = 2 if max(img.size) < 1200 else 1
                    if scale > 1:
                        img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
                    img = img.convert('L')
                    return np.array(img)

            # Use OpenCV for stronger preprocessing
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                # Fallback read
                with open(image_path, 'rb') as f:
                    data = np.frombuffer(f.read(), np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

            # Scale up small images
            h, w = img.shape[:2]
            if max(h, w) < 1200:
                scale = 1200.0 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Denoise while keeping edges
            gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

            # Adaptive threshold to handle dark mode UIs
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
            )

            # Morphological operations to strengthen characters
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

            return processed
        except Exception as e:
            logger.error(f"Error in _preprocess_for_easyocr: {e}")
            # Return simple grayscale as last resort
            with Image.open(image_path) as img:
                return np.array(img.convert('L'))

    def _extract_with_easyocr(self, image_path):
        """Attempt text extraction with EasyOCR. Returns (text, quality_score)."""
        if self._easyocr_reader is None:
            return None, 0.0
        try:
            preprocessed = self._preprocess_for_easyocr(image_path)
            result = self._easyocr_reader.readtext(preprocessed, detail=1, paragraph=True)
            texts = []
            confidences = []
            for item in result:
                try:
                    # item format: [bbox, text, confidence]
                    texts.append(item[1])
                    confidences.append(float(item[2]))
                except Exception:
                    continue
            combined_text = "\n".join(texts).strip()
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            logger.info(f"EasyOCR extracted {len(combined_text)} chars with avg_conf={avg_conf:.2f}")
            return combined_text, avg_conf
        except Exception as e:
            logger.error(f"EasyOCR extraction error: {e}")
            return None, 0.0

    def _extract_with_easyocr_tiled(self, image_path, tiles: int = 4):
        """Slice image into horizontal tiles and read with EasyOCR to improve recall for long chats."""
        if self._easyocr_reader is None:
            return None, 0.0
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                width, height = img.size
                step = max(1, height // tiles)
                texts = []
                confidences = []
                for top in range(0, height, step):
                    box = (0, top, width, min(height, top + step + 10))
                    crop = img.crop(box)
                    arr = np.array(crop.convert('L'))
                    try:
                        result = self._easyocr_reader.readtext(arr, detail=1, paragraph=True)
                        for item in result:
                            texts.append(item[1])
                            try:
                                confidences.append(float(item[2]))
                            except Exception:
                                pass
                    except Exception:
                        continue
                combined_text = "\n".join(texts).strip()
                avg_conf = float(np.mean(confidences)) if confidences else 0.0
                return combined_text, avg_conf
        except Exception as e:
            logger.error(f"EasyOCR tiled extraction error: {e}")
            return None, 0.0
    
    def _optimize_image(self, image_path):
        """Optimize image for faster processing and better text extraction"""
        try:
            # Open the image
            with Image.open(image_path) as img:
                # Apply image enhancement for better text clarity
                from PIL import ImageEnhance, ImageFilter
                
                # Convert to RGB if needed (handles RGBA, etc.)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Apply minimal enhancements - just enough for text clarity
                # Increase contrast slightly
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.3)
                
                # Resize large images to reduce processing time
                max_dimension = 1200  # Smaller size for faster processing
                if max(img.size) > max_dimension:
                    # Calculate new dimensions while maintaining aspect ratio
                    if img.width > img.height:
                        new_width = max_dimension
                        new_height = int(img.height * (max_dimension / img.width))
                    else:
                        new_height = max_dimension
                        new_width = int(img.width * (max_dimension / img.height))
                    
                    # Use BILINEAR for faster resizing (vs LANCZOS)
                    img = img.resize((new_width, new_height), Image.BILINEAR)
                    logger.info(f"Resized image from {img.width}x{img.height} to {new_width}x{new_height}")
                
                # Save to bytes with medium quality for faster processing
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85, optimize=True)
                buffer.seek(0)
                
                return buffer.getvalue()
        
        except Exception as e:
            logger.error(f"Error optimizing image: {str(e)}")
            # Return original image data if optimization fails
            with open(image_path, "rb") as f:
                return f.read()
    
    def _extract_text_fallback(self, image_path):
        """Fallback method for text extraction using more aggressive approach"""
        try:
            # Try a different image enhancement approach
            with Image.open(image_path) as img:
                # Convert to grayscale for better text contrast
                img = img.convert('L')
                
                # Apply threshold to make text more distinct
                from PIL import ImageOps
                img = ImageOps.autocontrast(img, cutoff=5)
                
                # Save to bytes
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                optimized_image = buffer.getvalue()
            
            # Encode optimized image to base64
            base64_image = base64.b64encode(optimized_image).decode('utf-8')
            
            # Very aggressive fallback prompt
            fallback_prompt = """
            This is an image that may contain hard-to-read text. Please examine it EXTREMELY carefully.
            
            I need you to extract ANY text visible in this image, even if:
            - The image is blurry, low resolution, or poor quality
            - The text is partially visible or cut off
            - The text is small or in the background
            
            Make your best attempt to read ANYTHING that looks like text.
            If you see partial words or letters, include them and make your best guess.
            This is likely a WhatsApp or messaging app screenshot.
            """
            
            # Allow more time for large images too
            timeout_seconds = 30
            
            # Call OpenAI Vision API with a different prompt and model configuration

            logger.info(f"Using fallback text extraction method (timeout: {timeout_seconds}s)")
            if self.using_v1:
                response = self.client.chat.completions.create(
                    model=self.vision_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a specialized OCR assistant with exceptional ability to read text from poor quality images. Your task is to extract ANY visible text, no matter how difficult it is to read."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": fallback_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=700,
                    temperature=0.2,

                    request_timeout=timeout_seconds
                )
                description = response.choices[0].message.content.strip()
            else:
                response = openai.ChatCompletion.create(
                    model=self.vision_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a specialized OCR assistant with exceptional ability to read text from poor quality images. Your task is to extract ANY visible text, no matter how difficult it is to read."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": fallback_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=700,
                    temperature=0.2,
                    request_timeout=timeout_seconds
                )
                description = response.choices[0].message.content.strip()

            logger.debug(f"Fallback text extraction result: {description[:100]}...")
            
            # Always return model's best attempt, even if it includes caveats
            if description and isinstance(description, str):
                return description.strip()
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Error in fallback text extraction: {str(e)}")
            return f"Error extracting text from image: {str(e)}"

    def summarize_image(self, image_path):
        """Send the image directly to the model to transcribe as much as possible and provide a concise summary.

        Returns a dict with keys: transcript, summary
        """
        if not os.path.exists(image_path):
            return {"transcript": "", "summary": "Error: Image file not found"}

        try:
            optimized_image = self._optimize_image(image_path)
            base64_image = base64.b64encode(optimized_image).decode('utf-8')

            instruction = (
                "You are reading a WhatsApp chat screenshot. This is CRITICAL - you must transcribe ALL text verbatim and understand the conversation context.\n\n"
                "MANDATORY: Transcribe EVERYTHING you can see, including:\n"
                "- All chat messages and conversations\n"
                "- Names of people in the chat\n"
                "- Banking details (IBAN, SWIFT codes, account numbers, bank names)\n"
                "- Company names and business information\n"
                "- Financial instructions or wire transfer details\n"
                "- Any embedded forms, cards, or documents with important information\n"
                "- Timestamps and message details\n"
                "- Any UI elements with text\n\n"
                "Then provide a detailed summary that shows you understand the conversation context. If you see specific banking information, mention it specifically. If some parts are unclear, make a best-effort reading and mark with [?]."
            )


            logger.info("Summarizing image content using vision API")
            if self.using_v1:
                response = self.client.chat.completions.create(
                    model=self.vision_model,
                    messages=[
                        {"role": "system", "content": instruction},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Transcribe and summarize this screenshot."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                            ],
                        },
                    ],
                    max_tokens=700,
                    temperature=0.3,
                )
                content = response.choices[0].message.content.strip()
            else:
                response = openai.ChatCompletion.create(
                    model=self.vision_model,
                    messages=[
                        {"role": "system", "content": instruction},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Transcribe and summarize this screenshot."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                            ],
                        },
                    ],
                    max_tokens=700,
                    temperature=0.3,
                )
                content = response.choices[0].message.content.strip()

            logger.debug(f"Image summary generated: {content[:100]}...")

            # Heuristically split transcript and summary
            transcript = content
            summary = ""
            # If the model separates with headings, try to parse
            lower = content.lower()
            if "summary:" in lower:
                parts = re.split(r"summary:\s*", content, flags=re.IGNORECASE)
                transcript = parts[0].strip()
                summary = parts[1].strip() if len(parts) > 1 else ""
            elif "transcript:" in lower:
                parts = re.split(r"transcript:\s*", content, flags=re.IGNORECASE)
                transcript = parts[-1].strip()
            return {"transcript": transcript, "summary": summary or transcript}
        except Exception as e:
            logger.error(f"summarize_image error: {e}")
            return {"transcript": "", "summary": f"Error summarizing image: {e}"}

    def _format_professional_reply(self, body: str, recipient_name: str | None = None) -> str:
        """Return the response as-is without any formatting or signatures."""
        try:
            content = (body or "").strip()
            # Remove any existing signatures or greetings
            content = re.sub(r'\n\nBest regards,?\s*.*$', '', content, flags=re.IGNORECASE)
            content = re.sub(r'\n\nSincerely,?\s*.*$', '', content, flags=re.IGNORECASE)
            content = re.sub(r'\n\nWarm regards,?\s*.*$', '', content, flags=re.IGNORECASE)
            content = re.sub(r'^Hello,?\s*[^,\n]*,?\s*\n\n', '', content, flags=re.IGNORECASE)
            content = re.sub(r'^Hi,?\s*[^,\n]*,?\s*\n\n', '', content, flags=re.IGNORECASE)
            content = re.sub(r'^Dear,?\s*[^,\n]*,?\s*\n\n', '', content, flags=re.IGNORECASE)
            return content.strip()
        except Exception:
            return body

    def generate_structured_reply(self, extracted_text: str, recipient_name: str | None = None) -> str:
        """
        Generate a balanced ChatGPT-like response:
        - Respond directly to the content
        - Be thorough but concise
        - Maintain natural conversation flow
        """
        instruction = """You are ChatGPT. Respond exactly like ChatGPT would respond to this message.

IMPORTANT INSTRUCTIONS:
1. Analyze the content thoroughly and provide a complete response
2. Include specific details from the message (names, companies, requirements)
3. Provide context and explanations where helpful
4. For business inquiries, give practical advice and options
5. Write 2-3 detailed paragraphs that thoroughly address the topic
6. Use a conversational, helpful tone like ChatGPT
7. NEVER include any signatures, greetings, or formalities like "Best regards" or "[Your Name]"
8. NEVER include placeholders like "[Your Position]" or "[Your Company]"
9. Just provide the direct, helpful response without any signature format
10. If the message mentions Adyen, payment processing, or business requirements, provide detailed information about options and requirements"""
        user_prompt = f"{extracted_text}"

        try:
            logger.info(f"Generating structured reply for text: {extracted_text[:100]}...")
            if self.using_v1:
                response = self.client.chat.completions.create(
                    model=self.text_model,
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=600,
                )
                result = response.choices[0].message.content.strip()
            else:
                response = openai.ChatCompletion.create(
                    model=self.text_model,
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=600,
                )
                result = response.choices[0].message.content.strip()
            logger.debug(f"Structured reply generated: {result[:100]}...")
            return result
        except Exception as e:
            logger.error(f"Error in generate_structured_reply: {e}")
            return "Summary:\n- (Could not generate)\n\nReply:\nSorry, I couldn't generate the reply."

    def approve_and_save(self, save_type, fields):
        if not save_type or not fields:
            return None, "Missing save type or fields"
            
        try:
            if save_type == 'Client_Log':
                result = self.airtable_client.create_record('Client_Log', fields)
            elif save_type == 'Templates':
                result = self.airtable_client.create_record('Templates', fields)
            elif save_type == 'Offers':
                result = self.airtable_client.create_record('Offers', fields)
            elif save_type == 'Knowledge_Base':   # ✅ New case
                result = self.airtable_client.create_record('Knowledge_Base', fields)
            else:
                return None, "Invalid save type"
                
            return (result, f"Successfully saved to {save_type}") if result else (None, f"Failed to save to {save_type}")
        except Exception as e:
            logger.error(f"Error in approve_and_save: {str(e)}")
            return None, f"Error: {str(e)}"


class AirtableClient:
    """Client for interacting with Airtable API"""
    
    def __init__(self):
        self.api_key = settings.AIRTABLE_API_KEY if hasattr(settings, 'AIRTABLE_API_KEY') else os.environ.get('AIRTABLE_API_KEY', '')
        self.base_id = settings.AIRTABLE_BASE_ID if hasattr(settings, 'AIRTABLE_BASE_ID') else os.environ.get('AIRTABLE_BASE_ID', '')
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        self.base_url = f'https://api.airtable.com/v0/{self.base_id}'
    
    def check_client_exists(self, contact_info, message_content=None):
        """Fast check if client exists in Clients or Leads tables"""
        if not contact_info and not message_content:
            return None, None

        if contact_info:
            # Direct filter query instead of fetching all records
            if '@' in contact_info:
                filter_formula = f"{{Email}} = '{contact_info.strip()}'"
            elif contact_info.isdigit() or contact_info.replace("+", "").isdigit():
                filter_formula = f"{{Phone}} = '{contact_info.strip()}'"
            else:
                filter_formula = f"{{Full_Name}} = '{contact_info.strip()}'"

            # First search Clients
            clients = self.get_table_records('Clients', filter_formula)
            if clients:
                return clients[0], 'Clients'

            # Then search Leads
            leads = self.get_table_records('Leads', filter_formula)
            if leads:
                return leads[0], 'Leads'

        # Optional message duplicate check (can also be optimized)
        if message_content:
            cleaned_message = self._normalize_text(message_content)
            all_leads = self.get_table_records('Leads')
            for lead in all_leads:
                fields = lead.get('fields', {})
                lead_message = fields.get('Message', '') or fields.get('Notes', '')
                norm_existing = self._normalize_text(lead_message)
                if self._is_duplicate_content(cleaned_message, norm_existing):
                    lead['is_message_duplicate'] = True
                    return lead, 'Leads'

        return None, None

    def _sanitize_fields_for_airtable(self, fields: dict) -> dict:
        """Clean common junk before sending to Airtable."""
        if not isinstance(fields, dict):
            return fields

        out = dict(fields)

        # 1) Clean message text
        if isinstance(out.get("Message"), str):
            msg = out["Message"].strip()
            # remove "plaintext" header that appears from UI
            msg = re.sub(r"(?i)^plaintext\s*\n+", "", msg).strip()

            # Drop 1–3 top UI lines typical of chat headers (group title, 'members', time-only lines, lone 'You/hey')
            lines = [ln for ln in msg.splitlines()]
            drop = 0
            for i in range(min(4, len(lines))):
                ln = lines[i].strip()
                if (re.search(r"\bmembers\b", ln, re.I) or
                    re.fullmatch(r"\d{1,2}:\d{2}\s*(AM|PM)?", ln, re.I) or
                    ln.lower() in {"you", "hey"} or
                    re.search(r"-\s*[^-]+$", ln)):  # e.g., "Nebroo LLC - Giovanni ..."
                    drop = i + 1
                else:
                    break
            if drop:
                msg = "\n".join(lines[drop:]).strip()

            out["Message"] = msg

        # 2) Sanitize Source field - only allow exact known values
        if "Source" in out:
            allowed_sources = {"WhatsApp", "Telegram", "Email"}
            if out["Source"] not in allowed_sources:
                logger.warning(f"Removing invalid Source value: {out['Source']}")
                del out["Source"]
                
        # 3) Remove Status field completely to avoid permission issues
        if "Status" in out:
            logger.warning(f"Removing Status field '{out['Status']}' to avoid permission issues")
            del out["Status"]


        return out


    def lead_exists(self, full_name=None, phone=None, email=None):
        """
        Strict duplicate check for Leads table.
        Returns existing lead if Full_Name OR Phone OR Email already exists.
        """
        filters = []

        if full_name:
            filters.append(f"{{Full_Name}} = '{full_name.strip()}'")
        if phone:
            filters.append(f"{{Phone}} = '{phone.strip()}'")
        if email:
            filters.append(f"{{Email}} = '{email.strip()}'")

        if not filters:
            return None

        # Build OR filter formula for Airtable
        filter_formula = "OR(" + ",".join(filters) + ")"

        existing = self.get_table_records('Leads', filter_formula)
        if existing:
            return existing[0]  # return the first matching record
        return None


    def _normalize_text(self, text):
        if not text:
            return ''
        t = text.lower()
        # Remove boilerplate extraction phrases
        t = re.sub(r"sure,? here'?s the (extracted )?text .*?image:?", ' ', t)
        t = re.sub(r"\btranscription:?\b", ' ', t)
        t = re.sub(r"\bmessages?:?\b", ' ', t)
        t = re.sub(r"\bheader:?\b", ' ', t)
        t = re.sub(r"from the image:?", ' ', t)
        # Remove timestamps and clock times
        t = re.sub(r"\b\d{1,2}:\d{2}\s*(am|pm)?\b", ' ', t)
        # Remove extra bullets/markdown
        t = t.replace('*', ' ').replace('`', ' ').replace('—', ' ').replace('–', ' ')
        # Collapse non-alphanumeric except @ and + (for emails/phones) to spaces
        t = re.sub(r"[^a-z0-9@+\s]", ' ', t)
        # Collapse whitespace
        t = re.sub(r"\s+", ' ', t).strip()
        return t

    def check_lead_message_duplicate(self, message_content, client_name=None):
        """Check if a lead with similar message content already exists.
        If client_name is provided, also check by name.
        """
        # First check by client name if provided
        if client_name:
            # Get all leads
            all_leads = self.get_table_records('Leads')
            for lead in all_leads:
                fields = lead.get('fields', {})
                lead_name = fields.get('Full_Name', '')
                # If names match, consider it a duplicate
                if lead_name and lead_name.lower() == client_name.lower():
                    logger.info(f"Found duplicate lead by name: {client_name}")
                    return True, lead
        
        # Then check by message content
        norm_new = self._normalize_text(message_content)
        if not norm_new:
            return False, None
            
        all_leads = self.get_table_records('Leads')
        for lead in all_leads:
            fields = lead.get('fields', {})
            lead_message = fields.get('Message', '') or fields.get('Notes', '')
            norm_existing = self._normalize_text(lead_message)
            if not norm_existing:
                continue
            # Use fuzzy comparison for robustness
            if self._is_duplicate_content(norm_new, norm_existing, threshold=0.7):
                logger.info(f"Found duplicate lead by content similarity")
                return True, lead
                
        return False, None
    
    def _is_duplicate_content(self, content1, content2, threshold=0.8):
        """Check if two content strings are duplicates based on similarity"""
        if not content1 or not content2:
            return False
        
        # Clean and normalize content
        content1_clean = content1.lower().strip()
        content2_clean = content2.lower().strip()
        
        # Exact match
        if content1_clean == content2_clean:
            return True
        
        # Check if one is contained in the other (for partial duplicates)
        if content1_clean in content2_clean or content2_clean in content1_clean:
            return True
        
        # Simple word overlap check
        words1 = set(content1_clean.split())
        words2 = set(content2_clean.split())
        
        if len(words1) > 0 and len(words2) > 0:
            common_words = words1.intersection(words2)
            similarity = len(common_words) / max(len(words1), len(words2))
            return similarity >= threshold
        
        return False

    def create_lead(self, name, contact_info, message=None):
        """Create a new lead in the Leads table with proper handling of name/phone/email"""
        if not contact_info and not name:
            return None

        fields = {}

        # ✅ अगर name proper है (और सिर्फ digits नहीं है)
        if name and not name.isdigit():
            fields['Full_Name'] = name
        else:
            # वरना contact_info check करो
            if contact_info:
                if '@' in contact_info:   # Email case
                    fields['Email'] = contact_info
                elif contact_info.isdigit() or contact_info.replace("+", "").isdigit():
                    fields['Phone'] = contact_info
                else:
                    fields['Full_Name'] = contact_info

        # Normalize and save message if available
        if message:
            cleaned_message = self._normalize_text(message)
            fields['Message'] = cleaned_message

        return self.create_record('Leads', fields)


    
    def log_message(self, client_id, client_table, message_content, response_content=None, source='Manual'):
        """Log a message in the Client_Log table"""
        if not client_id or not message_content:
            return None
            
        fields = {
            'Client': [client_id],  # Airtable linked record format
            'Message': message_content,
            'Source': source,
            'Type': 'Message'
        }
        
        if response_content:
            fields['Response'] = self._format_response_for_airtable(response_content)
            
        return self.create_record('Client_Log', fields)
    def save_to_knowledge_base(self, client_message, response):
        logger.info(f"save_to_kb called with type={type(response)}, value={response}")
        
        if not isinstance(response, dict):
            response = {"Final_Approved_Reply": str(response)}

        fields = dict(response)
        fields["Client_Message"] = client_message
        
        # Check for duplicate entries before saving
        existing_entry = self.check_duplicate_knowledge_base(client_message, fields.get("Final_Approved_Reply", ""))
        if existing_entry:
            logger.info(f"🔍 Duplicate Knowledge Base entry found: {existing_entry.get('id')}")
            return {
                'id': existing_entry.get('id'),
                'duplicate': True,
                'message': 'This response already exists in Knowledge Base'
            }
        
        logger.info(f"🔍 Final fields going to create_record: {fields}")
        return self.create_record("Knowledge_Base", fields)

    def check_duplicate_knowledge_base(self, client_message, final_response):
        """Check if a similar entry already exists in Knowledge Base"""
        try:
            # Get all knowledge base entries
            knowledge_entries = self.get_table_records('Knowledge_Base')
            if not knowledge_entries:
                return None
            
            client_message_lower = client_message.lower().strip()
            final_response_lower = final_response.lower().strip()
            
            for entry in knowledge_entries:
                fields = entry.get('fields', {})
                existing_message = fields.get('Client_Message', '').lower().strip()
                existing_response = fields.get('Final_Approved_Reply', '').lower().strip()
                
                # Check for exact matches or very similar content
                if (existing_message == client_message_lower and 
                    existing_response == final_response_lower):
                    return entry
                
                # Check for high similarity (80%+ match)
                if (self._calculate_similarity(existing_message, client_message_lower) > 0.8 and
                    self._calculate_similarity(existing_response, final_response_lower) > 0.8):
                    return entry
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking duplicate knowledge base: {e}")
            return None

    def _calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts (0-1 scale)"""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

    def find_knowledge_base_response(self, query):
        """Search Knowledge_Base table for matching response"""
        if not query or query.strip() == "":
            return None
            
        # Get all knowledge base entries
        knowledge_entries = self.get_table_records('Knowledge_Base')
        if not knowledge_entries:
            return None
            
        best_match = None
        best_score = 0
        
        # Simple text matching algorithm
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for entry in knowledge_entries:
            fields = entry.get('fields', {})
            question = fields.get('Client_Message', '')
            answer = fields.get('Final_Approved_Reply', '')
            
            if not question or not answer:
                continue
                
            # Direct match - return immediately
            if query_lower == question.lower():
                return {
                    'content': answer,
                    'id': entry.get('id')
                }
            
            # Check for partial match - but be more strict
            # Only allow partial matches if the shorter string is at least 80% of the longer string
            shorter = min(query_lower, question.lower())
            longer = max(query_lower, question.lower())
            
            if shorter in longer and len(shorter) >= len(longer) * 0.8:
                return {
                    'content': answer,
                    'id': entry.get('id')
                }
                
            # Word overlap scoring - only for substantial matches
            question_words = set(question.lower().split())
            common_words = query_words.intersection(question_words)
            
            # Calculate score based on word overlap percentage
            if len(query_words) > 0 and len(question_words) > 0:
                overlap_score = len(common_words) / max(len(query_words), len(question_words))
                
                # Only consider matches with high overlap (80%+) and at least 2 common words
                if overlap_score > best_score and overlap_score >= 0.8 and len(common_words) >= 2:
                    best_score = overlap_score
                    best_match = {
                        'content': answer,
                        'id': entry.get('id')
                    }
        
        # Return best match if it exceeds threshold
        if best_score > 0.6:  # At least 60% word overlap
            return best_match
                
        return None
    
    def get_table_records(self, table_name, filters=None):
        """Get records from a specific table with optional filters"""
        url = f"{self.base_url}/{table_name}"
        
        params = {}
        if filters:
            if isinstance(filters, str):
                # already a valid Airtable formula like "OR({Email}='x', {Phone}='y')"
                params['filterByFormula'] = filters
            elif isinstance(filters, dict):
                # build from dict -> AND({Field}='Value', ...)
                params['filterByFormula'] = self._build_filter_formula(filters)
            else:
                logger.warning(f"Unsupported filters type: {type(filters)} - ignoring filters")

        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                return response.json().get('records', [])
            else:
                logger.error(f"Airtable API Error: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting records from Airtable: {str(e)}")
            return []

    
    def _build_filter_formula(self, filters):
        """Build Airtable filter formula from filters dict"""
        conditions = []
        for field, value in filters.items():
            conditions.append(f"{{{field}}} = '{value}'")
        
        if len(conditions) == 1:
            return conditions[0]
        else:
            return f"AND({','.join(conditions)})"
    
    def create_record(self, table_name, fields):
        url = f"{self.base_url}/{table_name}"

        # Debug log to check what is going
        logger.info(f"📌 create_record: table={table_name}, type={type(fields)}, fields={fields}")

        # Always enforce dict
        if not isinstance(fields, dict):
            raise TypeError(f"Expected dict for fields, got {type(fields)}")

        # Sanitize fields for Airtable constraints (e.g., valid select options)
        safe_fields = self._sanitize_fields_for_airtable(fields)
        payload = {"fields": safe_fields}

        # ✅ Use json= instead of data=
        resp = requests.post(url, headers=self.headers, json=payload)

        if resp.status_code >= 400:
            logger.error(f"Airtable error {resp.status_code}: {resp.text}")
            
            # Handle specific error cases
            if resp.status_code == 429:
                error_data = resp.json()
                if 'errors' in error_data and len(error_data['errors']) > 0:
                    error_msg = error_data['errors'][0].get('message', 'API billing limit exceeded')
                    logger.error(f"Airtable API billing limit exceeded: {error_msg}")
                    # Return a structured error response
                    return {
                        'error': True,
                        'status_code': 429,
                        'message': 'Airtable API billing limit exceeded. Please upgrade your plan or wait for the limit to reset.',
                        'details': error_msg
                    }
            elif resp.status_code == 422:
                logger.error(f"Airtable validation error: {resp.text}")
                return {
                    'error': True,
                    'status_code': 422,
                    'message': 'Airtable validation error. Please check your data.',
                    'details': resp.text
                }

        return resp.json()



    def create_record_with_fallback(self, table_name, fields):
        """Attempt to create a record; if it fails (e.g., unknown field names or single-select issues),
        retry with common alternative field names.
        """
        # First try as-is
        result = self.create_record(table_name, fields)
        if result:
            return result

        # Build variants with alternate field names
        key_variants = {
            'Full_Name': ['Full_Name', 'Full Name', 'Name'],
            'Phone': ['Phone', 'Phone Number', 'Mobile', 'Contact', 'Contact_Number'],
            'Email': ['Email', 'E-mail', 'Email Address'],
            'Message': ['Message', 'Description'],
            'Source': ['Source'],
            'Status': ['Status'],
        }

        # Generate a small set of reasonable permutations
        from itertools import product
        base = {
            'Full_Name': fields.get('Full_Name'),
            'Phone': fields.get('Phone'),
            'Email': fields.get('Email'),
            'Message': fields.get('Message'),
            'Source': fields.get('Source'),
            'Status': fields.get('Status'),
        }

        variants_lists = []
        for key, options in key_variants.items():
            value = base.get(key)
            if value is None:
                variants_lists.append([None])
            else:
                variants_lists.append([(opt, value) for opt in options])

        # Create combinations only for keys that have values
        combos = []
        for combo in product(*variants_lists):
            variant = {}
            for item in combo:
                if item and item[1] is not None:
                    variant[item[0]] = item[1]
            if variant:
                combos.append(variant)

        # Ensure original order first, then variants
        tried = set()
        for variant_fields in [fields] + combos[:8]:  # cap to avoid too many attempts
            sig = tuple(sorted(variant_fields.keys()))
            if sig in tried:
                continue
            tried.add(sig)
            result = self.create_record(table_name, variant_fields)
            if result:
                return result

        return None

    def _build_search_filters(self, search_term):
        """Build multiple search filters to increase chance of finding a match"""
        filters = []
        
        # Clean up the search term
        search_term = search_term.strip()
        
        # Direct matches
        if '@' in search_term:  # Looks like an email
            filters.append({'Email': search_term})
        
        # Phone number search (try different formats)
        phone_clean = ''.join(filter(str.isdigit, search_term))
        if len(phone_clean) >= 7:  # Minimum length for a phone number
            filters.append({'Phone': search_term})
            # Try with just the digits
            if phone_clean != search_term:
                filters.append({'Phone': phone_clean})
        
        # Name search
        if len(search_term.split()) >= 2:  # Might be a full name
            filters.append({'Name': search_term})
        
        # Add a general search filter for partial matches
        if len(search_term) > 3:  # Avoid very short terms
            # Note: This is a simplified approach - Airtable's actual search capabilities may vary
            filters.append({'Name': f"FIND('{search_term}')"})
        
        return filters

    def get_client_details(self, client_id, table_name):
        """Get detailed client information including history"""
        if not client_id or not table_name:
            return None
        
        # Get client record
        client = None
        try:
            url = f"{self.base_url}/{table_name}/{client_id}"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                client = response.json()
            else:
                logger.error(f"Error getting client details: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Exception getting client details: {str(e)}")
            return None
        
        if not client:
            return None
        
        # Get client history from Client_Log
        client_history = []
        try:
            # Use FILTERBYFORMULA to find logs for this client
            formula = f"Client='{client_id}'"
            url = f"{self.base_url}/Client_Log?filterByFormula={formula}"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                logs = response.json().get('records', [])
                client_history = sorted(logs, key=lambda x: x.get('createdTime', ''), reverse=True)
            else:
                logger.error(f"Error getting client history: {response.status_code}")
        except Exception as e:
            logger.error(f"Exception getting client history: {str(e)}")
        
        # Combine client info with history
        result = {
            'client': client,
            'history': client_history,
            'table': table_name
        }
        
        return result 

    def _format_professional_reply(self, body: str, recipient_name: str | None = None) -> str:
        """Ensure the reply starts with Hello and ends with Best regards signature."""
        try:
            content = (body or "").strip()
            greeting_name = (recipient_name or "there").strip()
            # Prepend greeting if missing
            lower_head = content[:30].lower()
            if not any(x in lower_head for x in ["hello", "dear", "hi "]):
                content = f"Hello, {greeting_name},\n\n" + content
            # Append sign-off if missing
            lower_tail = content[-120:].lower()
            if not any(x in lower_tail for x in ["regards", "sincerely", "best regards", "warm regards"]):
                content = content.rstrip() + f"\n\nBest Regards,\n{self.sender_name}"
            return content
        except Exception:
            return body

    def generate_structured_reply(self, extracted_text: str, recipient_name: str | None = None) -> str:
        """Compose a concise professional reply based on extracted screenshot text.
        Always starts with Hello, and ends with Best Regards + sender name.
        """
        instruction = (
            "Write a concise, professional response to the client's message. "
            "Start with 'Hello, {name}' (use 'there' if name unknown). "
            "Be clear and helpful and avoid extra disclaimers. End with 'Best regards, {sender}'."
        )
        try:

            logger.info(f"Generating professional structured reply for text: {extracted_text[:100]}...")
            if self.using_v1:
                response = self.client.chat.completions.create(
                    model=self.text_model,
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": f"Client message:\n{extracted_text}\n\nReply:"}
                    ],
                    temperature=0.3,
                    max_tokens=350,
                )
                draft = response.choices[0].message.content.strip()
            else:
                response = openai.ChatCompletion.create(
                    model=self.text_model,
                    messages=[
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": f"Client message:\n{extracted_text}\n\nReply:"}
                    ],
                    temperature=0.3,
                    max_tokens=350,
                )
                draft = response.choices[0].message.content.strip()

            logger.debug(f"Professional structured reply generated: {draft[:100]}...")
        except Exception as e:
            logger.error(f"generate_structured_reply error: {e}")
            draft = "Thank you for reaching out. We will review your message and get back to you shortly."
        return self._format_professional_reply(draft, recipient_name)

    def search_knowledge_base(self, query, threshold=0.6):
        """Enhanced search in Knowledge_Base table for matching responses"""
        if not query or query.strip() == "":
            return []
            
        # Get all knowledge base entries
        knowledge_entries = self.get_table_records('Knowledge_Base')
        if not knowledge_entries:
            return []
            
        matches = []
        
        # Enhanced text matching algorithm
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for entry in knowledge_entries:
            fields = entry.get('fields', {})
            question = fields.get('Client_Message', '')
            answer = fields.get('Final_Approved_Reply', '')
            category = fields.get('Category', '')
            
            if not question or not answer:
                continue
                
            # Direct match - highest priority
            if query_lower == question.lower():
                matches.append({
                    'content': answer,
                    'id': entry.get('id'),
                    'question': question,
                    'score': 1.0,
                    'category': category
                })
                continue
                
            # Partial match - check if query is contained in question
            if query_lower in question.lower():
                matches.append({
                    'content': answer,
                    'id': entry.get('id'),
                    'question': question,
                    'score': 0.9,
                    'category': category
                })
                continue
                
            # Word overlap scoring
            question_words = set(question.lower().split())
            common_words = query_words.intersection(question_words)
            
            # Calculate score based on word overlap percentage
            if len(query_words) > 0 and len(question_words) > 0:
                overlap_score = len(common_words) / max(len(query_words), len(question_words))
                
                # Check if score exceeds threshold
                if overlap_score > threshold:
                    matches.append({
                        'content': answer,
                        'id': entry.get('id'),
                        'question': question,
                        'score': overlap_score,
                        'category': category
                    })
        
        # Sort matches by score (highest first)
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        return matches 

    def search_accounts(self, query, threshold=0.6):
        """Search Accounts table for records matching the query in any field."""
        if not query or query.strip() == "":
            return []
        
        # Get all account entries
        account_entries = self.get_table_records('Accounts')
        if not account_entries:
            return []
        
        matches = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for entry in account_entries:
            fields = entry.get('fields', {})
            # Combine all field values into a single string for matching
            combined = ' '.join(str(v) for v in fields.values() if v)
            combined_lower = combined.lower()
            
            # Direct match
            if query_lower == combined_lower:
                matches.append({
                    'fields': fields,
                    'id': entry.get('id'),
                    'score': 1.0
                })
                continue
            # Partial match
            if query_lower in combined_lower:
                matches.append({
                    'fields': fields,
                    'id': entry.get('id'),
                    'score': 0.9
                })
                continue
            # Word overlap
            combined_words = set(combined_lower.split())
            common_words = query_words.intersection(combined_words)
            if len(query_words) > 0 and len(combined_words) > 0:
                overlap_score = len(common_words) / max(len(query_words), len(combined_words))
                if overlap_score > threshold:
                    matches.append({
                        'fields': fields,
                        'id': entry.get('id'),
                        'score': overlap_score
                    })
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches


    def search_clients(self, query, threshold=0.6):
        """Search Clients table for records matching the query in any field."""
        if not query or query.strip() == "":
            return []

        # Get all client entries
        client_entries = self.get_table_records('Clients')
        if not client_entries:
            return []

        matches = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for entry in client_entries:
            fields = entry.get('fields', {})
            # Combine all field values into a single string for matching
            combined = ' '.join(str(v) for v in fields.values() if v)
            combined_lower = combined.lower()

            # Direct match
            if query_lower == combined_lower:
                matches.append({
                    'fields': fields,
                    'id': entry.get('id'),
                    'score': 1.0
                })
                continue
            # Partial match
            if query_lower in combined_lower:
                matches.append({
                    'fields': fields,
                    'id': entry.get('id'),
                    'score': 0.9
                })
                continue
            # Word overlap
            combined_words = set(combined_lower.split())
            common_words = query_words.intersection(combined_words)
            if len(query_words) > 0 and len(combined_words) > 0:
                overlap_score = len(common_words) / max(len(query_words), len(combined_words))
                if overlap_score > threshold:
                    matches.append({
                        'fields': fields,
                        'id': entry.get('id'),
                        'score': overlap_score
                    })
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches
    def search_clients_by_contact(self, contact):
        """Search Clients table by contact information (phone/email)"""
        if not contact:
            return None
        
        # Search by email
        if '@' in contact:
            email_filter = f"{{Email}} = '{contact.strip()}'"
            clients = self.get_table_records('Clients', email_filter)
            if clients:
                return clients[0]
        
        # Search by phone
        if contact.replace("+", "").replace("-", "").replace(" ", "").isdigit():
            phone_filter = f"{{Phone_Formatted}} = '{contact.strip()}'"
            clients = self.get_table_records('Clients', phone_filter)
            if clients:
                return clients[0]
        
        return None

    def search_clients_by_name(self, name):
        """Search Clients table by name"""
        if not name:
            return None
        
        name_filter = f"{{Full_Name}} = '{name.strip()}'"
        clients = self.get_table_records('Clients', name_filter)
        if clients:
            return clients[0]
        
        return None

    def search_accounts_by_name(self, company_name):
        """Search Accounts table by company name"""
        if not company_name:
            return None
        
        company_filter = f"{{Company_Name}} = '{company_name.strip()}'"
        accounts = self.get_table_records('Accounts', company_filter)
        if accounts:
            return accounts[0]
        
        return None

    def create_client_record(self, name, contact, message):
        """Create a new client record in Clients table"""
        if not name or name.lower() in ['unknown', 'n/a', 'na', 'none']:
            logger.warning(f"Not creating client record - invalid name: {name}")
            return None
            
        # Ensure name is properly formatted
        fields = {
            'Full_Name': name.strip(),
            'Status': 'Active'
        }
        
        if contact:
            if '@' in contact:
                fields['Email'] = contact
            elif contact.replace("+", "").replace("-", "").replace(" ", "").isdigit():
                fields['Phone_Formatted'] = contact
        
        if message:
            fields['Notes'] = message[:500]  # Limit notes length
        
        # Try to create record, log any failures
        try:
            result = self.create_record('Clients', fields)
            if not result or (isinstance(result, dict) and result.get('error')):
                logger.error(f"Failed to create client record: {result}")
                # Try with fallback field names
                result = self.create_record_with_fallback('Clients', fields)
            return result
        except Exception as e:
            logger.error(f"Exception creating client record: {str(e)}")
            return None

    def detect_source_from_content(self, content):
        """Detect the likely source (WhatsApp, Email, Telegram) from message content."""
        if not content:
            return None
        
        content_lower = content.lower()
        
        # WhatsApp indicators
        if any(x in content_lower for x in ["whatsapp", "last seen", "typing…", "online", "wa.me/"]):
            return "WhatsApp"
        
        # Telegram indicators
        if any(x in content_lower for x in ["telegram", "t.me/", "forwarded message", "telegram channel"]):
            return "Telegram"
        
        # Email indicators
        if any(x in content_lower for x in ["subject:", "from:", "to:", "cc:", "bcc:", "forwarded message", "@gmail.com", "@outlook.com", "@yahoo.com"]):
            return "Email"
        
        # Default - don't set any source if we can't detect it
        return None
    def create_lead_record(self, name, contact, message, source=None, final_response=None):
        """Create a new lead record in Leads table.
        If name is not provided (or is an unknown placeholder), omit the Full_Name field.
        """
        fields = {}
        
        # CRITICAL: Always prioritize final_response over message
        if final_response and isinstance(final_response, str) and final_response.strip():
            fields['Message'] = final_response.strip()
            # Log for debugging
            logger.info(f"Using final_response for Message field: {final_response[:100] if final_response else 'None'}")
        elif message and isinstance(message, str) and message.strip():
            fields['Message'] = message.strip()
            logger.info(f"Using original message for Message field (no final_response provided)")
        else:
            logger.warning("No message content available for lead record")
            fields['Message'] = "No content available"
        
        # Do not set Status field at all - let Airtable use its default
        # fields['Status'] = "Open"  # Removed as it causes permission issues
        
        # Try to detect source from content if not provided
        if not source:
            source = self.detect_source_from_content(fields['Message'])
        # Only add Source if we can detect it with high confidence
        # Otherwise, don't set it to avoid permission issues
        if source in {"WhatsApp", "Telegram", "Email"}:
            fields['Source'] = source
        # Don't set default "Other" as it might cause permission issues

        # Only include Full_Name if it's a real, valid name (not a timestamp or placeholder)
        if name and isinstance(name, str):
            trimmed = name.strip()
            # Check if name looks like a timestamp (e.g., "1:30") or other invalid name
            is_timestamp = bool(re.match(r'^\d{1,2}:\d{2}$', trimmed))
            is_placeholder = trimmed.lower() in ['unknown', 'n/a', 'na', 'none']
            
            # Only add name if it's valid
            if trimmed and not is_timestamp and not is_placeholder:
                fields['Full_Name'] = trimmed
            else:
                logger.info(f"Skipping invalid name: '{trimmed}'")
                # Don't set Full_Name at all for invalid names
        
        if contact:
            if '@' in contact:
                fields['Email'] = contact
            elif contact.replace("+", "").replace("-", "").replace(" ", "").isdigit():
                fields['Phone'] = contact
        
        return self.create_record('Leads', fields)

    def search_leads(self, query, threshold=0.6):
        """Search Leads table for records matching the query in any field."""
        if not query or query.strip() == "":
            return []
        
        # Get all lead entries
        lead_entries = self.get_table_records('Leads')
        if not lead_entries:
            return []
        
        matches = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for entry in lead_entries:
            fields = entry.get('fields', {})
            # Combine all field values into a single string for matching
            combined = ' '.join(str(v) for v in fields.values() if v)
            combined_lower = combined.lower()
            
            # Direct match
            if query_lower == combined_lower:
                matches.append({
                    'fields': fields,
                    'id': entry.get('id'),
                    'score': 1.0
                })
                continue
            # Partial match
            if query_lower in combined_lower:
                matches.append({
                    'fields': fields,
                    'id': entry.get('id'),
                    'score': 0.9
                })
                continue
            # Word overlap
            combined_words = set(combined_lower.split())
            common_words = query_words.intersection(combined_words)
            if len(query_words) > 0 and len(combined_words) > 0:
                overlap_score = len(common_words) / max(len(query_words), len(combined_words))
                if overlap_score > threshold:
                    matches.append({
                        'fields': fields,
                        'id': entry.get('id'),
                        'score': overlap_score
                    })
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches

    def update_record(self, table_name, record_id, fields):
        """Update an existing record in Airtable"""
        url = f"{self.base_url}/{table_name}/{record_id}"
        
        payload = {"fields": fields}
        
        try:
            response = requests.patch(url, headers=self.headers, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Airtable update error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error updating record in Airtable: {str(e)}")
            return None

    def _format_response_for_airtable(self, response):
        """Format response text for Airtable in a professional email-like format"""
        if not response:
            return ""
            
        # Check if the response already has a subject line
        if response.startswith("Subject:"):
            return response
            
        # Format the response with a standard subject line and signature
        formatted_response = "Subject: Assistance with Your Recent Inquiry\n\n"
        
        # Add salutation if not present
        if not any(greeting in response[:50].lower() for greeting in ["dear", "hello", "hi ", "greetings"]):
            formatted_response += "Dear Client,\n\n"
            
        # Add the main content
        formatted_response += response.strip()
        
        # Add signature if not present
        if not any(closing in response[-100:].lower() for closing in ["regards", "sincerely", "thank you", "best", "warm"]):
            formatted_response += "\n\nWarm regards,\n\n[Your Name]\n[Your Position]\n[Your Contact Information]"
            
        return formatted_response 

def generate_image_hash(image_file):
        hasher = hashlib.sha256()
        for chunk in image_file.chunks():
            hasher.update(chunk)
        return hasher.hexdigest()
