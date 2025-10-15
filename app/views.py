import os
import logging
import json
import io
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages as django_messages
from django.views.generic import ListView, CreateView, UpdateView, DetailView
from django.urls import reverse_lazy
from django.http import JsonResponse, HttpResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.http import Http404
import re
import unicodedata
import requests
from django.conf import settings
from django.http import JsonResponse
from django.utils import timezone
from .models import Message

from .models import Message, OpenAIConfig
from .forms import MessageForm, ResponseEditForm
from .utils import AIClient, AirtableClient

import numpy as np
from PIL import Image
import easyocr
import openai
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.core.files.uploadedfile import InMemoryUploadedFile

# Configure logging
logger = logging.getLogger(__name__)

def optimize_image_for_ocr(input_path, max_side=1000, jpeg_quality=75):
    """Resize/compress large images to a manageable size for OCR to avoid timeouts.
    Uses smaller dimensions and quality for faster processing.
    """
    try:
        try:
            file_size = os.path.getsize(input_path)
        except Exception:
            file_size = 0
        with Image.open(input_path) as im:
            im = im.convert('RGB')
            width, height = im.size
            # Smaller max_side (1200 vs 1800) and lower size threshold for faster processing
            if max(width, height) <= max_side and file_size <= 1048576:  # 1MB threshold
                return input_path, None
            scale = max_side / float(max(width, height))
            new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            # Use BILINEAR instead of LANCZOS for faster resize
            im = im.resize(new_size, Image.BILINEAR)
            from tempfile import NamedTemporaryFile
            tmp = NamedTemporaryFile(delete=False, suffix=".jpg")
            im.save(tmp.name, format='JPEG', quality=jpeg_quality, optimize=True)
            return tmp.name, tmp.name
    except Exception:
        return input_path, None

@csrf_exempt
@require_POST
def update_message(request, message_id):
    try:
        message = Message.objects.get(id=message_id)
        data = json.loads(request.body)
        new_content = data.get('content')
        
        if new_content:
            message.content = new_content
            message.save()
            return JsonResponse({'status': 'success'})
        else:
            return JsonResponse({'status': 'error', 'message': 'No content provided'}, status=400)
            
    except Message.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Message not found'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

# Dashboard View
class DashboardView(ListView):
    model = Message
    template_name = 'app/dashboard.html'
    context_object_name = 'messages'
    ordering = ['-created_at']
    
    def get_queryset(self):
        # Show all messages for sidebar/history
        return Message.objects.all().order_by('-created_at')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = MessageForm()
        # Get selected message by GET param
        message_id = self.request.GET.get('message_id')
        selected_message = None
        if message_id:
            try:
                selected_message = Message.objects.get(id=message_id)
            except Message.DoesNotExist:
                selected_message = None
        else:

        #     # Default: latest message
            selected_message = Message.objects.order_by('-created_at').first()
        context['selected_message'] = selected_message
        return context

# Message Creation View
class MessageCreateView(CreateView):
    model = Message
    form_class = MessageForm
    template_name = 'app/dashboard.html'
    success_url = reverse_lazy('dashboard')
    
    def form_valid(self, form):
        message = form.save(commit=False)
        
        # Initialize AI client
        ai_client = AIClient()
        
        # Process message based on content or image
        if message.content:
            # Extract client information from text
            client_name, client_contact = ai_client.extract_client_info(message.content)
            if client_name:
                message.client_name = client_name
            if client_contact:
                message.client_contact = client_contact
                
            # Detect language and generate AI response
            detected_language = detect_language(message.content)
            response_data = ai_client.generate_response(message.content, recipient_name=client_name, detected_language=detected_language)
            message.ai_response = response_data['content']
            message.from_knowledge_base = response_data['from_knowledge_base']
            message.knowledge_base_id = response_data['knowledge_base_id']
            message.status = Message.PROCESSED
            # Set final_response to ai_response initially
            message.final_response = message.ai_response
            
        elif message.image:
            # Extract text from image
            try:
                # Save the message first to ensure the image is saved to disk
                message.status = "pending"
                message.ai_response = "Processing your image... This may take a moment."
                message.save()
                
                # Set the message ID in the session so we can display it
                self.request.session['new_message_id'] = message.id
                
                # Display immediate feedback to the user
                # django_messages.info(self.request, "Processing your image... Please wait a moment.")
                
                # Now get the image path
                image_path = message.image.path
                logger.info(f"Processing image at path: {image_path}")
                
                # Check if the image file exists
                if not os.path.exists(image_path):
                    logger.error(f"Image file not found: {image_path}")
                    message.ai_response = "Error: The uploaded image file could not be found."
                    message.extracted_content = "Failed to extract text from image"
                    message.status = Message.PROCESSED
                    message.save()
                    # django_messages.error(self.request, "Error: The uploaded image file could not be found.")
                    return redirect('dashboard')            
            # Extract text from image (optimize first to avoid 502/timeouts on large uploads)
                optimized_path, _cleanup_path = optimize_image_for_ocr(image_path)
                try:
                    extracted_text = ai_client.extract_text_from_image(optimized_path)
                finally:
                    if _cleanup_path:
                        try:
                            os.unlink(_cleanup_path)
                        except Exception:
                            pass
                
                # If text extraction is weak or fails, try image summarization
                if not extracted_text or extracted_text.startswith("Error") or len(extracted_text.strip()) < 40:
                    logger.info("Text extraction weak, trying image summarization...")
                    summary_obj = ai_client.summarize_image(optimized_path)
                    extracted_text = summary_obj.get('transcript', '') or summary_obj.get('summary', '')
                    logger.info(f"Image summarization result: {extracted_text[:100]}...")
                
                if extracted_text and len(extracted_text.strip()) > 10:
                    message.extracted_content = extracted_text
                    
                    logger.info(f"Successfully extracted/analyzed image content")
                    
                    # Extract client information from extracted text
                    client_name, client_contact = ai_client.extract_client_info(extracted_text)
                    
                    # Check for specific names in the text that might be missed by the AI
                    if "Chiara Fazzini" in extracted_text and not client_name:
                        client_name = "Chiara Fazzini"
                        logger.info(f"Forced client name to '{client_name}' based on text content")
                    
                    if client_name:
                        message.client_name = client_name
                        logger.info(f"Setting client name: {client_name}")
                    if client_contact:
                        message.client_contact = client_contact
                        logger.info(f"Setting client contact: {client_contact}")
                    
                    # Compose a structured professional reply from extracted text
                    try:
                        logger.info(f"Extracted text length: {len(extracted_text)}")
                        logger.info(f"Extracted text preview: {extracted_text[:200]}...")
                        logger.info(f"Client name: {client_name}")
                        
                        if hasattr(ai_client, 'generate_structured_reply'):
                            logger.info("Using generate_structured_reply for image processing")
                            message.ai_response = ai_client.generate_structured_reply(extracted_text, client_name)
                            logger.info(f"Generated response preview: {message.ai_response[:200]}...")
                        else:
                            logger.info("Using fallback generate_response for image processing")
                            # Fallback: use standard generator with greeting/signoff
                            resp = ai_client.generate_response(
                                f"Please draft a concise professional reply to this message:\n\n{extracted_text}",
                                is_followup=True,
                                recipient_name=client_name,
                            )
                            message.ai_response = resp.get('content', extracted_text)
                    except Exception as e:
                        logger.error(f"Error generating structured reply: {e}")
                        # Last resort: keep extracted text
                        message.ai_response = extracted_text
                    message.from_knowledge_base = False
                    message.knowledge_base_id = None
                    # Set final_response to the extracted text
                    message.final_response = message.ai_response
                    # django_messages.success(self.request, "Image processed successfully!")

                    # Auto-save based on presence of client name
                    try:
                        airtable_client = AirtableClient()
                        raw_message = (message.extracted_content or message.content or "Image upload").strip()
                        client_name = (message.client_name or "").strip()
                        client_contact = (message.client_contact or "").strip()

                        if client_name:
                            # Ensure Client exists
                            existing_client = airtable_client.search_clients_by_name(client_name)
                            if not existing_client:
                                airtable_client.create_client_record(client_name, client_contact, raw_message)
                            # Save to Knowledge Base
                            kb_result = airtable_client.save_to_knowledge_base(raw_message, message.final_response or message.ai_response)
                            if kb_result and isinstance(kb_result, dict) and not kb_result.get('error'):
                                if not kb_result.get('duplicate'):
                                    message.saved_to_knowledge_base = True
                                    message.knowledge_base_id = kb_result.get('id')
                            # Create Lead with name
                            lead_result = airtable_client.create_lead_record(client_name, client_contact, raw_message)
                            if lead_result and isinstance(lead_result, dict) and lead_result.get('id'):
                                message.saved_to_airtable = True
                                message.airtable_lead_id = lead_result.get('id')
                        else:
                            # No name: save only messages to KB and Leads (no Full_Name)
                            kb_result = airtable_client.save_to_knowledge_base(raw_message, message.final_response or message.ai_response)
                            if kb_result and isinstance(kb_result, dict) and not kb_result.get('error') and not kb_result.get('duplicate'):
                                message.saved_to_knowledge_base = True
                                message.knowledge_base_id = kb_result.get('id')
                            # Pass final response to save in lead with source detection
                            lead_result = airtable_client.create_lead_record(
                                None, 
                                client_contact, 
                                raw_message,
                                source=None,  # Auto-detect from content
                                final_response=message.final_response or message.ai_response
                            )
                            if lead_result and isinstance(lead_result, dict) and lead_result.get('id'):
                                message.saved_to_airtable = True
                                message.airtable_lead_id = lead_result.get('id')
                    except Exception as e:
                        logger.error(f"Airtable autosave error: {e}")
                else:
                    message.ai_response = "I couldn't extract readable text from your image. Please try uploading a clearer image or type your message directly."
                    message.extracted_content = "No text could be extracted"
                    # django_messages.warning(self.request, "No text could be extracted from your image.")
                    
                message.status = Message.PROCESSED
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                message.ai_response = f"An error occurred while processing your image: {str(e)}. Please try again or type your message directly."
                message.status = Message.PROCESSED
                # django_messages.error(self.request, "An error occurred while processing your image.")
        
        # Save the message
        message.save()
        
        # Set the message ID in the session so we can display it
        self.request.session['new_message_id'] = message.id
        
        # Clear any previous messages from the session to avoid displaying them
        if 'previous_messages' in self.request.session:
            del self.request.session['previous_messages']
        
        # Remove success notification
        # django_messages.success(self.request, "AI response generated successfully!")
        
        return redirect('dashboard')
    
    def form_invalid(self, form):
        error_message = "Please provide either a message OR upload a screenshot."
        django_messages.error(self.request, error_message)
        return super().form_invalid(form)

# Message Update View (for editing responses)
class MessageUpdateView(UpdateView):
    model = Message
    form_class = ResponseEditForm
    template_name = 'app/message_edit.html'
    success_url = reverse_lazy('dashboard')
    
    def get_object(self, queryset=None):
        try:
            return super().get_object(queryset)
        except Http404:
            # If the message doesn't exist, redirect to dashboard
            django_messages.error(self.request, "The message you're trying to edit doesn't exist or has been deleted.")
            # Don't return redirect here, as it causes the error
            # Instead, raise Http404 to trigger the proper error handling
            raise Http404("Message does not exist")
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Pre-populate edited_response with ai_response if it's empty
        if not self.object.edited_response:
            context['form'].initial['edited_response'] = self.object.ai_response
        
        # Get client information if available
        client_info = None
        if self.object.client_contact:
            # Initialize Airtable client
            airtable_client = AirtableClient()
            
            # Check if client exists
            client, table = airtable_client.check_client_exists(self.object.client_contact)
            if client and table:
                # Get detailed client information
                client_info = airtable_client.get_client_details(client.get('id'), table)
        
        context['client_info'] = client_info
        
        # Alternative responses section removed as requested
        
        return context
    
    def form_valid(self, form):
        try:
            message = form.save(commit=False)
            
            # Always set final_response to edited_response or ai_response
            if message.edited_response and message.edited_response.strip():
                message.final_response = message.edited_response
            else:
                message.final_response = message.ai_response
                
            message.status = Message.APPROVED
            message.save()
            
            # Make sure this message is shown on the dashboard
            self.request.session['new_message_id'] = message.id
            
            django_messages.success(self.request, "Response updated successfully!")
            return super().form_valid(form)
        except Exception as e:
            logger.error(f"Error updating message: {str(e)}")
            django_messages.error(self.request, f"Error updating response: {str(e)}")
            return self.form_invalid(form)

@require_POST
def save_to_knowledge_base(request, pk):
    """
    Complete workflow implementation:
    1. Check if client exists in Clients table
    2. If exists, show popup with options to save or update
    3. If not exists, show popup to create new client
    4. Save response to Knowledge_Base
    5. Create/update Leads record
    """
    message = get_object_or_404(Message, pk=pk)
    response_content = message.final_response or message.ai_response
    if not response_content:
        return JsonResponse({'success': False, 'message': 'No response to save.'}, status=400)

    try:
        airtable_client = AirtableClient()

        # Get client information
        contact = (message.client_contact or "").strip()
        client_name = (message.client_name or "").strip()
        raw_message = (message.extracted_content or message.content or "Image upload").strip()

        # Search for existing client in Clients table
        existing_client = None
        existing_account = None
        
        if contact:
            # Search by contact info
            existing_client = airtable_client.search_clients_by_contact(contact)
        
        if client_name and not existing_client:
            # Search by name if no contact match
            existing_client = airtable_client.search_clients_by_name(client_name)
        
        # Search for associated account
        if existing_client:
            account_name = existing_client.get('fields', {}).get('Associated_Account', '')
            if account_name:
                existing_account = airtable_client.search_accounts_by_name(account_name)

        if existing_client:
            # Client exists - show popup with options
            return JsonResponse({
                'success': True,
                'client_exists': True,
                'client_data': existing_client,
                'account_data': existing_account,
                'message': 'Client found in database. Choose an option:',
                'options': {
                    'save_response': 'Save response to Knowledge Base',
                    'update_client': 'Update client information',
                    'create_lead': 'Create new lead entry'
                }
            })
        else:
            # Client doesn't exist - show popup to create new
            return JsonResponse({
                'success': True,
                'client_exists': False,
                'message': 'Client not found. Create new client?',
                'client_info': {
                    'name': client_name,
                    'contact': contact,
                    'message': raw_message
                }
            })

    except Exception as e:
        logger.error(f"save_to_knowledge_base error: {e}")
        return JsonResponse({'success': False, 'message': str(e)}, status=500)


@require_POST
def handle_client_action(request, pk):
    """Handle actions from the client popup (save, update, create)"""
    message = get_object_or_404(Message, pk=pk)
    data = json.loads(request.body)
    action = data.get('action')
    
    try:
        airtable_client = AirtableClient()
        response_content = message.final_response or message.ai_response
        raw_message = (message.extracted_content or message.content or "Image upload").strip()
        
        # If frontend sends an updated response (edited text), use it and persist
        response_override = data.get('response_override')
        if response_override and isinstance(response_override, str) and response_override.strip():
            response_content = response_override.strip()
            # Persist the latest edited content so future actions use it
            message.final_response = response_content
            try:
                message.save()
            except Exception as _e:
                logger.warning(f"Failed to persist response_override: {_e}")
        
        if action == 'save_response':
            # 1) Save response to Knowledge Base (idempotent)
            kb_payload = {
                "Client_Message": raw_message,
                "Final_Approved_Reply": response_content
            }
            kb_result = airtable_client.save_to_knowledge_base(kb_payload["Client_Message"], kb_payload)

            # Handle explicit billing limit error
            if kb_result and isinstance(kb_result, dict) and kb_result.get('error') and kb_result.get('status_code') == 429:
                return JsonResponse({
                    'success': False,
                    'message': 'Airtable API billing limit exceeded. Please upgrade your Airtable plan or wait for the limit to reset.',
                    'error_type': 'billing_limit'
                })

            kb_saved = False
            kb_duplicate = False
            if kb_result:
                if kb_result.get('duplicate'):
                    kb_duplicate = True
                else:
                    kb_saved = True
                    message.saved_to_knowledge_base = True
                    message.knowledge_base_id = kb_result.get('id')

            # 2) Always create a Lead (with or without name) - but check for duplicates first
            client_name = (message.client_name or '').strip() or None
            client_contact = (message.client_contact or '').strip() or None
            
            # Check for duplicate lead by name or content
            is_duplicate, existing_lead = airtable_client.check_lead_message_duplicate(raw_message, client_name)
            if is_duplicate:
                logger.info(f"Skipping lead creation - duplicate found: {existing_lead}")
                lead_result = existing_lead
                lead_saved = True
            else:
                # 3) Also create a Client record if client_name exists
                if client_name:
                    client_result = airtable_client.create_client_record(client_name, client_contact, raw_message)
                    logger.info(f"Created client record: {client_result}")
                    
                try:
                    # CRITICAL: Ensure we have a valid response_content to save
                    if not response_content or not response_content.strip():
                        logger.warning("Empty response_content detected, using raw_message as fallback")
                        response_content = raw_message
                    
                    # Log the actual content we're saving to help with debugging
                    logger.info(f"Saving to Leads with final_response: {response_content[:100] if response_content else 'None'}")
                    
                    # Pass ONLY the final response to save in the lead
                    lead_result = airtable_client.create_lead_record(
                        client_name, 
                        client_contact, 
                        None,  # Raw message not needed since we're using final_response
                        source=None,  # Auto-detect from content
                        final_response=response_content  # This should be the Final_Approved_Reply content
                    )
                except Exception as _e:
                    lead_result = None
                    logger.error(f"Lead creation error: {_e}")

            if not is_duplicate:
                lead_saved = False
                if lead_result and isinstance(lead_result, dict) and lead_result.get('id'):
                    lead_saved = True
                    message.saved_to_airtable = True
                    message.airtable_lead_id = lead_result.get('id')
            else:
                # If we found a duplicate, mark it as saved
                if existing_lead and isinstance(existing_lead, dict) and existing_lead.get('id'):
                    message.saved_to_airtable = True
                    message.airtable_lead_id = existing_lead.get('id')

            message.save()

            # Build user-friendly message
            msg_parts = []
            if kb_saved:
                msg_parts.append('Knowledge Base saved')
            elif kb_duplicate:
                msg_parts.append('KB duplicate (skipped)')
            else:
                msg_parts.append('KB save failed')

            msg_parts.append('Lead created' if lead_saved else 'Lead creation failed')

            return JsonResponse({
                'success': kb_saved or kb_duplicate or lead_saved,
                'message': '; '.join(msg_parts)
            })
        
        elif action == 'create_new_client':
            # Create new client and lead
            client_name = data.get('name', message.client_name or 'Unknown')
            client_contact = data.get('contact', message.client_contact or '')
            
            # Create client record
            client_result = airtable_client.create_client_record(client_name, client_contact, raw_message)
            
            # Create lead record
            lead_result = airtable_client.create_lead_record(client_name, client_contact, raw_message)
            
            # Save to Knowledge Base
            payload = {
                "Client_Message": raw_message,
                "Final_Approved_Reply": response_content
            }
            kb_result = airtable_client.save_to_knowledge_base(payload["Client_Message"], payload)
            
            if client_result and lead_result and kb_result:
                if kb_result.get('duplicate'):
                    return JsonResponse({
                        'success': False, 
                        'message': 'This response already exists in Knowledge Base. Client and Lead created, but no duplicate response saved.',
                        'duplicate': True
                    })
                else:
                    message.saved_to_knowledge_base = True
                    message.knowledge_base_id = kb_result.get('id')
                    message.saved_to_airtable = True
                    message.airtable_lead_id = lead_result.get('id')
                    message.save()
                    
                    return JsonResponse({
                        'success': True,
                        'message': f'New client "{client_name}" created successfully! Client, Lead, and Knowledge Base records updated.'
                    })
            else:
                return JsonResponse({
                    'success': False,
                    'message': 'Some records were created but there were errors. Please check Airtable.'
                })
        
        elif action == 'update_existing_client':
            # Update existing client information
            client_id = data.get('client_id')
            update_data = data.get('update_data', {})
            
            # Update client record
            update_result = airtable_client.update_record('Clients', client_id, update_data)
            
            # Save to Knowledge Base
            payload = {
                "Client_Message": raw_message,
                "Final_Approved_Reply": response_content
            }
            kb_result = airtable_client.save_to_knowledge_base(payload["Client_Message"], payload)
            
            if update_result and kb_result:
                if kb_result.get('duplicate'):
                    return JsonResponse({
                        'success': False, 
                        'message': 'This response already exists in Knowledge Base. Client updated, but no duplicate response saved.',
                        'duplicate': True
                    })
                else:
                    message.saved_to_knowledge_base = True
                    message.knowledge_base_id = kb_result.get('id')
                    message.save()
                    
                    return JsonResponse({
                        'success': True,
                        'message': 'Client information updated and response saved to Knowledge Base!'
                    })
            else:
                return JsonResponse({
                    'success': False,
                    'message': 'Failed to update client or save to Knowledge Base'
                })
        
        else:
            return JsonResponse({
                'success': False,
                'message': 'Invalid action specified'
            })
            
    except Exception as e:
        logger.error(f"handle_client_action error: {e}")
        return JsonResponse({'success': False, 'message': str(e)}, status=500)


@require_POST
def search_clients(request):
    """Search across Clients, Accounts, and Leads tables"""
    try:
        data = json.loads(request.body)
        query = data.get('query', '').strip()
        
        if not query:
            return JsonResponse({'success': False, 'message': 'Search query is required'})
        
        airtable_client = AirtableClient()
        
        # Search across all tables
        results = {
            'clients': [],
            'accounts': [],
            'leads': []
        }
        
        # Search Clients table
        try:
            clients = airtable_client.search_clients(query)
            results['clients'] = clients[:10]  # Limit to 10 results
        except Exception as e:
            logger.error(f"Error searching clients: {e}")
        
        # Search Accounts table
        try:
            accounts = airtable_client.search_accounts(query)
            results['accounts'] = accounts[:10]  # Limit to 10 results
        except Exception as e:
            logger.error(f"Error searching accounts: {e}")
        
        # Search Leads table
        try:
            leads = airtable_client.search_leads(query)
            results['leads'] = leads[:10]  # Limit to 10 results
        except Exception as e:
            logger.error(f"Error searching leads: {e}")
        
        return JsonResponse({
            'success': True,
            'results': results,
            'query': query
        })
        
    except Exception as e:
        logger.error(f"search_clients error: {e}")
        return JsonResponse({'success': False, 'message': str(e)}, status=500)

# Create Lead in Airtable
@require_POST
def create_lead(request, pk):
    message = get_object_or_404(Message, pk=pk)
    
    if not message.client_name or not message.client_contact:
        return JsonResponse({
            'success': False,
            'message': 'Missing client name or contact information'
        })
    
    try:
        airtable_client = AirtableClient()
        
        # Normalize message content for duplicate check
        message_content = message.content or message.extracted_content or "Image upload"
        # Check for duplicate by message or name in Leads
        is_duplicate, existing_lead = airtable_client.check_lead_message_duplicate(message_content, message.client_name)
        if is_duplicate:
            return JsonResponse({
                'success': False,
                'message': 'This lead already exists',
                'is_duplicate': True,
                'existing_lead': existing_lead.get('id') if existing_lead else None
            })
        # Check for duplicate by contact info (phone/email) in Leads
        client, table = airtable_client.check_client_exists(message.client_contact, message_content)
        if table == 'Leads' and client:
            return JsonResponse({
                'success': False,
                'message': 'This lead already exists',
                'is_duplicate': True,
            })
        else:
            # Create new lead
            result = airtable_client.create_lead(
                message.client_name,
                message.client_contact,
                message.content or message.extracted_content or "Image upload"
            )
            
            if result:
                message.saved_to_airtable = True
                message.airtable_lead_id = result.get('id')
                message.save()
                
                # Log message for the new lead
                airtable_client.log_message(
                    result.get('id'),
                    'Leads',
                    message.content or message.extracted_content or "Image upload",
                    message.final_response or message.ai_response
                )
                
                return JsonResponse({
                    'success': True,
                    'message': 'New lead created in Airtable'
                })
            else:
                return JsonResponse({
                    'success': False,
                    'message': 'Failed to create lead in Airtable'
                })
    except Exception as e:
        logger.error(f"Error creating lead: {str(e)}")
        return JsonResponse({
            'success': False,
            'message': f'Error: {str(e)}'
        })

def detect_telegram_data(text):
    if "Telegram" in text or "Digital Nomads" in text:  # or any keyword unique to Telegram
        source = "Telegram"
        
        # Extract name
        name_match = re.search(r'(?<=\n)(\w+):|(?<=\n)(\w+)\sjoined the group', text)
        name = name_match.group(1) if name_match else None

        # Extract contact (Telegram username usually not shown, skip or use default)
        contact = None  # or default like "Telegram User"

        return {
            "source": source,
            "name": name or "Telegram User",
            "contact": contact
        }


# Approve and Save Document/Template/Offer
@csrf_exempt
@require_POST
def approve_and_save(request):
    """
    Endpoint to approve and save a document/log/offer/template after user confirmation.
    Expects POST data: {'type': 'Client_Log'|'Templates'|'Offers', 'fields': {...}}
    """
    try:
        data = json.loads(request.body)
        save_type = data.get('type')
        fields = data.get('fields', {})
        
        if not save_type or not fields:
            return JsonResponse({
                'success': False,
                'message': 'Missing required parameters'
            })
        
        ai_client = AIClient()
        result, message = ai_client.approve_and_save(save_type, fields)
        
        if result:
            return JsonResponse({
                'success': True,
                'message': message,
                'id': result.get('id')
            })
        else:
            return JsonResponse({
                'success': False,
                'message': message
            })
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'message': 'Invalid JSON data'
        })
    except Exception as e:
        logger.error(f"Error in approve_and_save: {str(e)}")
        return JsonResponse({
            'success': False,
            'message': f'Error: {str(e)}'
        })

# Message Detail View
class MessageDetailView(DetailView):
    model = Message
    template_name = 'app/message_detail.html'
    context_object_name = 'message'
    
    def get_object(self, queryset=None):
        try:
            return super().get_object(queryset)
        except Http404:
            # If the message doesn't exist, redirect to dashboard
            django_messages.error(self.request, "The message you're trying to view doesn't exist or has been deleted.")
            return redirect('dashboard')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Add response form for editing
        message = self.object
        response_form = ResponseEditForm(instance=message)
        context['response_form'] = response_form
        
        # Get client information if available
        client_info = None
        if message.client_contact:
            # Initialize Airtable client
            airtable_client = AirtableClient()
            
            # Check if client exists
            client, table = airtable_client.check_client_exists(message.client_contact)
            if client and table:
                # Get detailed client information
                client_info = airtable_client.get_client_details(client.get('id'), table)
        
        context['client_info'] = client_info
        
        # Create response object for template
        if message.from_knowledge_base:
            response_type = 'knowledge_base'
        else:
            response_type = 'ai_generated'
            
        context['response'] = {
            'content': message.final_response or message.ai_response,
            'response_type': response_type,
            'airtable_knowledge_id': message.knowledge_base_id
        }
        
        # Add AI response for new messages
        
# Prefer the AI response (final_response if available, else ai_response)
        if message.final_response:
            fields['Message'] = message.final_response
        elif message.ai_response:
            fields['Message'] = message.ai_response   # yahan galti thi
        elif message.extracted_content:
            fields['Message'] = message.extracted_content


        
        return context

@csrf_exempt
@require_POST
def delete_message(request, message_id):
    """Delete a specific message"""
    try:
        message = get_object_or_404(Message, id=message_id)
        message.delete()
        # Always return JSON for AJAX/JS
        return JsonResponse({'success': True})
    except Exception as e:
        logger.error(f"Error deleting message: {str(e)}")
        return JsonResponse({'success': False, 'error': str(e)})

@require_POST
def response_create(request, message_id):
    """Create or update a response for a message"""
    message = get_object_or_404(Message, id=message_id)
    
    try:
        # Get form data
        content = request.POST.get('content')
        is_approved = request.POST.get('is_approved') == 'on'
        save_to_kb = 'save_to_kb' in request.POST
        response_type = request.POST.get('response_type')
        airtable_id = request.POST.get('airtable_id')
        create_lead = request.POST.get('create_lead') == 'on'
        
        # Update message with content
        if content:
            message.final_response = content
            logger.info(f"Updated final_response for message {message_id}: {content[:50]}...")
            
            # Analyze response if needed (e.g., extract additional info)
            ai_client = AIClient()
            # You can add analysis here if needed
        else:
            logger.warning(f"No content provided for message {message_id}")
            django_messages.warning(request, "No response content provided.")
            return redirect('message_detail', pk=message_id)
            
        # Save to knowledge base if approved or explicitly requested
        if is_approved or save_to_kb:
            airtable_client = AirtableClient()
            result = airtable_client.save_to_knowledge_base(
                message.content or message.extracted_content or "Image upload",
                message.final_response
            )
            
            if result:
                if result.get('duplicate'):
                    django_messages.warning(request, "This response already exists in Knowledge Base. No duplicate saved.")
                    logger.info(f"Duplicate Knowledge Base entry detected for message {message_id}")
                else:
                    message.saved_to_knowledge_base = True
                    message.knowledge_base_id = result.get('id')
                    django_messages.success(request, "Response saved to knowledge base.")
                    logger.info(f"Saved message {message_id} to knowledge base: {result.get('id')}")
            else:
                django_messages.error(request, "Failed to save to knowledge base.")
                logger.error(f"Failed to save message {message_id} to knowledge base")
        
        # Create lead if requested
        if create_lead and message.client_contact:
            airtable_client = AirtableClient()
            
            # Check if client already exists (including content duplicates)
            message_content = message.content or message.extracted_content or "Image upload"
            client, table = airtable_client.check_client_exists(message.client_contact, message_content)
            
            if client:
                # Client exists - show duplicate message
                django_messages.warning(request, f"This lead already exists in {table}.")
                logger.info(f"Duplicate lead detected for message {message_id} - client exists in {table}")
            else:
                # Create new lead
                result = airtable_client.create_lead(
                    message.client_name or "Unknown Client",
                    message.client_contact,
                    message.content or message.extracted_content or "Image upload"
                )
                
                if result:
                    message.saved_to_airtable = True
                    message.airtable_lead_id = result.get('id')
                    
                    # Log message for the new lead
                    log_result = airtable_client.log_message(
                        result.get('id'),
                        'Leads',
                        message.content or message.extracted_content or "Image upload",
                        message.final_response
                    )
                    
                    if log_result:
                        django_messages.success(request, "New lead created in Airtable and message logged.")
                        logger.info(f"Created lead and logged message {message_id} in Airtable: {result.get('id')}")
                    else:
                        django_messages.warning(request, "Lead created but failed to log message.")
                        logger.warning(f"Created lead but failed to log message {message_id}")
                else:
                    django_messages.error(request, "Failed to create lead in Airtable.")
                    logger.error(f"Failed to create lead for message {message_id}")
        
        # Save message
        message.status = Message.APPROVED
        message.save()
        logger.info(f"Message {message_id} saved with status APPROVED")
        
        # Set the message ID in the session so we can display it
        request.session['new_message_id'] = message.id
        
        return redirect('dashboard')
    except Exception as e:
        logger.error(f"Error creating response: {str(e)}")
        django_messages.error(request, f"Error creating response: {str(e)}")
        return redirect('message_detail', pk=message_id)

@require_POST
def process_response(request, message_id):
    """Process a response when the Send button is clicked"""
    message = get_object_or_404(Message, id=message_id)
    
    try:
        # Get form data
        content = request.POST.get('content')
        
        if not content:
            return JsonResponse({
                'success': False,
                'message': 'No content provided'
            })
        
        # Initialize AI client
        ai_client = AIClient()
        
        # Check for special commands
        content_lower = content.lower()
        special_command = False
        
        # No special commands needed - use standard OpenAI response generation
        special_command = False
        
        # If not a special command, generate response using OpenAI
        if not special_command:
            logger.info(f"Generating response for message {message_id}")
            
            # Generate context for the AI
            context = f"Original question: {message.content or message.extracted_content or 'Image upload'}\n"
            context += f"User input: {content}\n"
            context += f"Instructions: Generate a professional, well-formatted response. Use markdown formatting with bold headers, bullet points, and numbered lists where appropriate."
            
            # Generate new AI response
            response_data = ai_client.generate_response(context, is_followup=True)
            new_response = response_data['content']
            
        # Update the message with the new response
        message.final_response = new_response
        logger.info(f"Generated new response: {new_response[:50]}...")
        
        # Save the message
        message.save()
        
        # Return JSON response
        return JsonResponse({
            'success': True,
            'response': new_response
        })
    except Exception as e:
        logger.error(f"Error processing response: {str(e)}")
        return JsonResponse({
            'success': False,
            'message': str(e)
        })

def generate_pdf(request, message_id):
    """Generate a PDF from a message response"""
    message = get_object_or_404(Message, id=message_id)
    
    # Get the content
    content = message.final_response or message.ai_response
    if not content:
        django_messages.error(request, "No content to generate PDF from.")
        return redirect('message_detail', pk=message_id)
    
    # Create a file-like buffer to receive PDF data
    buffer = io.BytesIO()
    
    # Create the PDF document using ReportLab
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=72)
    
    # Create styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    title_style.alignment = 1  # Center alignment
    
    normal_style = styles['Normal']
    normal_style.fontSize = 12
    normal_style.leading = 14
    
    # Create a list to hold our document content
    elements = []
    
    # Add title
    if message.content:
        title = message.content
        if len(title) > 50:
            title = title[:50] + "..."
    else:
        title = "AI Generated Response"
    
    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(1, 0.5*inch))
    
    # Process the content to handle line breaks and formatting
    paragraphs = content.split('\n\n')
    for para in paragraphs:
        if para.strip():
            # Check if this is a list item
            if para.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '0.', 'Ã¢â‚¬Â¢', '-', '*')):
                elements.append(Paragraph(para, styles['Normal']))
            else:
                elements.append(Paragraph(para, normal_style))
            elements.append(Spacer(1, 0.2*inch))
    
    # Add footer
    footer_style = ParagraphStyle(
        'footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.gray
    )
    elements.append(Spacer(1, 0.5*inch))
    elements.append(Paragraph("Generated by AI Assistant", footer_style))
    
    # Build the PDF
    doc.build(elements)
    
    # Get the value of the BytesIO buffer and write it to the response
    pdf = buffer.getvalue()
    buffer.close()
    
    # Create the HttpResponse with PDF content
    response = HttpResponse(content_type='application/pdf')
    
    # Set the filename
    filename = f"response_{message_id}.pdf"
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    
    response.write(pdf)
    return response

@csrf_exempt
@require_POST
def ajax_enhance_response(request, pk):
    """AJAX endpoint to enhance a response with a follow-up using OpenAI"""
    import json
    try:
        message = Message.objects.get(pk=pk)
    except Message.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Message not found.'}, status=404)

    data = json.loads(request.body.decode('utf-8'))
    edited_response = data.get('edited_response', '')

    # Get OpenAI config from DB or fallback to settings
    config = OpenAIConfig.objects.filter(is_active=True).order_by('-updated_at').first()
    api_key = config.api_key if config else getattr(settings, 'OPENAI_API_KEY', None)
    model = config.model if config else getattr(settings, 'OPENAI_MODEL', 'gpt-3.5-turbo')
    temperature = config.temperature if config else 0.7
    max_tokens = config.max_tokens if config else 1000

    if not api_key:
        return JsonResponse({'success': False, 'error': 'OpenAI API key not configured.'}, status=400)

    prompt = edited_response.strip()

    # Call OpenAI
    try:
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = response['choices'][0]['message']['content']
    except Exception as e:
        return JsonResponse({'success': False, 'error': f'OpenAI error: {str(e)}'}, status=500)

    # Optionally, update the message object
    message.edited_response = edited_response
    message.final_response = answer
    # Optionally, append to conversation history
    if not message.conversation_history:
        message.conversation_history = []
    message.conversation_history.append({
        'user': prompt,
        'assistant': answer
    })
    message.save()

    return JsonResponse({'success': True, 'answer': answer})

# --- Chat History API for Sidebar ---
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

@csrf_exempt
def chat_history_api(request):
    if request.method == 'GET':
        messages = Message.objects.all().order_by('-created_at')[:50]
        data = [
            {
                'id': m.id,
                'content': m.content or m.extracted_content or 'Image',
                'created_at': m.created_at.strftime('%Y-%m-%d %H:%M'),
            }
            for m in messages
        ]
        return JsonResponse({'history': data})

def chat_detail_api(request, pk):
    try:
        m = Message.objects.get(pk=pk)
        data = {
            "id": m.id,
            "content": m.content or m.extracted_content or "Image",
            "ai_response": m.final_response or m.ai_response,
            "created_at": m.created_at.strftime('%Y-%m-%d %H:%M'),
        }
        return JsonResponse({"success": True, "chat": data})
    except Message.DoesNotExist:
        return JsonResponse({"success": False, "error": "Not found"}, status=404)

def get_message_json(request, pk):
    """AJAX endpoint to return a message's content and response as JSON, including image URL if present"""
    try:
        m = Message.objects.get(pk=pk)
        data = {
            "id": m.id,
            "content": m.content or m.extracted_content or "Image",
            "ai_response": m.final_response or m.ai_response,
            "created_at": m.created_at.strftime('%Y-%m-%d %H:%M'),
            "image_url": m.image.url if m.image else None,
        }
        return JsonResponse({"success": True, "message": data})
    except Message.DoesNotExist:
        return JsonResponse({"success": False, "error": "Not found"}, status=404)



@csrf_exempt
def save_to_airtable_accounts(request, message_id):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': 'Invalid request method.'}, status=405)

    try:
        message = Message.objects.get(id=message_id)
    except Message.DoesNotExist:
        return JsonResponse({'success': False, 'message': 'Message not found.'}, status=404)

    airtable_api_key = settings.AIRTABLE_API_KEY
    airtable_base_id = settings.AIRTABLE_BASE_ID
    table_name = 'Accounts'

    headers = {
        'Authorization': f'Bearer {airtable_api_key}',
        'Content-Type': 'application/json',
    }

    # Get message content
    message_text = getattr(message, 'final_response', '') or getattr(message, 'ai_response', '') or message.content or ''

    # Function to extract company names from text dynamically
    def extract_companies(text):
        companies = set()
        processed_positions = set()  # Track positions to avoid overlapping matches
        
        # Look for companies mentioned with "**" (bold formatting in markdown) - this is the main pattern
        bold_pattern = r'\*\*([^*]+?)\*\*'
        bold_matches = re.finditer(bold_pattern, text)
        for match in bold_matches:
            company = match.group(1).strip()
            start_pos = match.start()
            end_pos = match.end()
            
            # Skip if this position has already been processed
            if any(start_pos >= pos[0] and end_pos <= pos[1] for pos in processed_positions):
                continue
                
            # Filter out message labels and generic phrases, but allow real company names
            if (len(company.split()) >= 1 and 
                not any(generic in company.lower() for generic in [
                    'an american', 'an indian', 'the indian', 'a leading', 'another leading', 
                    'part of', 'as of', 'the latest', 'here are', 'with a significant',
                    'message to send', 'chat renamed', 'subject', 'dear', 'hi ', 'hello',
                    'hope this', 'looking forward', 'please feel', 'reach out', 'provide you',
                    'all the details', 'options available', 'certainly', 'here is', 'refined version'
                ]) and
                # Must contain company indicators or be a multi-word name
                (any(word in company.lower() for word in ['llc', 'ltd', 'inc', 'corp', 'company', 'international', 'solutions', 'technologies', 'systems', 'consulting', 'services', 'group', 'enterprises', 'holdings', 'partners']) or
                 len(company.split()) >= 2)):
                companies.add(company)
                processed_positions.add((start_pos, end_pos))
        
        # Look for companies with Limited/Ltd/Corp suffixes (only if not already found in bold)
        company_pattern = r'([A-Z][A-Za-z\s]+(?:Limited|Ltd|Corporation|Corp|Inc|LLC))'
        matches = re.finditer(company_pattern, text)
        for match in matches:
            company = match.group(1).strip()
            start_pos = match.start()
            end_pos = match.end()
            
            # Skip if this position overlaps with already processed bold text
            if any(start_pos >= pos[0] and end_pos <= pos[1] for pos in processed_positions):
                continue
                
            if len(company.split()) >= 2:
                companies.add(company)
                processed_positions.add((start_pos, end_pos))
        
        # Look for numbered lists that might contain company names (only if not already found)
        numbered_pattern = r'\d+\.\s*\*\*([^*]+?)\*\*'
        numbered_matches = re.finditer(numbered_pattern, text)
        for match in numbered_matches:
            company = match.group(1).strip()
            start_pos = match.start()
            end_pos = match.end()
            
            # Skip if this position overlaps with already processed text
            if any(start_pos >= pos[0] and end_pos <= pos[1] for pos in processed_positions):
                continue
                
            # Filter out generic phrases
            if (len(company.split()) >= 1 and 
                not any(generic in company.lower() for generic in ['an american', 'an indian', 'the indian', 'a leading', 'another leading', 'part of', 'as of'])):
                companies.add(company)
                processed_positions.add((start_pos, end_pos))
        
        # Additional pattern to catch company names that might be in parentheses (only if not already found)
        parenthetical_pattern = r'\*\*([^*]+?)\s*\(([^)]+)\)\*\*'
        parenthetical_matches = re.finditer(parenthetical_pattern, text)
        for match in parenthetical_matches:
            company = match.group(1).strip()
            abbreviation = match.group(2).strip()
            # Combine company name with abbreviation
            full_company = f"{company} ({abbreviation})"
            start_pos = match.start()
            end_pos = match.end()
            
            # Skip if this position overlaps with already processed text
            if any(start_pos >= pos[0] and end_pos <= pos[1] for pos in processed_positions):
                continue
                
            if (len(company.split()) >= 1 and 
                not any(generic in company.lower() for generic in ['an american', 'an indian', 'the indian', 'a leading', 'another leading', 'part of', 'as of'])):
                companies.add(full_company)
                processed_positions.add((start_pos, end_pos))
        
        return list(companies)

    # Step 1: Fetch existing Airtable records
    all_records = []
    url = f'https://api.airtable.com/v0/{airtable_base_id}/{table_name}'
    offset = None
    while True:
        params = {'offset': offset} if offset else {}
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            return JsonResponse({'success': False, 'message': 'Failed to fetch Airtable records.'}, status=500)

        data = response.json()
        all_records.extend(data.get('records', []))
        offset = data.get('offset')
        if not offset:
            break

    # Extract companies from message text
    companies = extract_companies(message_text)
    
    # If no companies found in the text, try to find them in the response
    if not companies and message.final_response:
        companies.extend(extract_companies(message.final_response))
    if not companies and message.ai_response:
        companies.extend(extract_companies(message.ai_response))

    # Remove duplicates while preserving order
    companies = list(dict.fromkeys(companies))

    # Check for existing companies
    existing_companies = []
    existing_company_names = []
    
    for record in all_records:
        company_name = record.get('fields', {}).get('Company_Name', '')
        if company_name:
            existing_company_names.append(company_name.lower())
    
    # Check which companies already exist
    for company_name in companies:
        if company_name.lower() in existing_company_names:
            existing_companies.append(company_name)
    
    # If all companies already exist, return early with duplicate message
    if existing_companies and len(existing_companies) == len(companies):
        return JsonResponse({
            'success': False,
            'message': 'This company already exists',
            'companies_found': companies,
            'existing_companies': existing_companies,
            'is_duplicate': True
        })
    
    # Filter out existing companies and only save new ones
    new_companies = [company for company in companies if company.lower() not in existing_company_names]
    
    # Save each new company
    saved_records = []
    for company_name in new_companies:
        new_record = {
            "Company_Name": company_name,
            "Type": "",
            "State_of_Formation": "",
            "EIN": "",
            "Associated_Client": message.client_contact or "",
            "Clients": "",
            "Status": "active"
        }
        saved_records.append({"company": company_name, "record": new_record})

    # Save records to Airtable
    post_url = f'https://api.airtable.com/v0/{airtable_base_id}/{table_name}'
    save_results = []
    
    for record in saved_records:
        post_data = {
            "fields": record["record"]
        }
        try:
            post_resp = requests.post(post_url, headers=headers, json=post_data)
            success = post_resp.status_code in [200, 201]
            
            # Handle specific error cases
            if post_resp.status_code == 429:
                error_data = post_resp.json()
                if 'errors' in error_data and len(error_data['errors']) > 0:
                    error_msg = error_data['errors'][0].get('message', 'API billing limit exceeded')
                    logger.error(f"Airtable API billing limit exceeded for company {record['company']}: {error_msg}")
                    save_results.append({
                        'company': record["company"],
                        'success': False,
                        'response': f"API billing limit exceeded: {error_msg}",
                        'error_type': 'billing_limit'
                    })
                    continue
            
            save_results.append({
                'company': record["company"],
                'success': success,
                'response': post_resp.text
            })
            logger.info(f"Saved company {record['company']} to Airtable: {success}")
        except Exception as e:
            logger.error(f"Error saving company {record['company']}: {str(e)}")
            save_results.append({
                'company': record["company"],
                'success': False,
                'error': str(e)
            })

    # Prepare response
    successful_saves = [r for r in save_results if r['success']]
    billing_limit_errors = [r for r in save_results if r.get('error_type') == 'billing_limit']
    
    if not companies:
        return JsonResponse({
            'success': False,
            'message': 'No company names found in the message',
            'companies_found': []
        })
    
    # Check for billing limit errors
    if billing_limit_errors:
        return JsonResponse({
            'success': False,
            'message': 'Airtable API billing limit exceeded. Please upgrade your Airtable plan or wait for the limit to reset.',
            'error_type': 'billing_limit',
            'companies_found': companies,
            'save_results': save_results
        })
    
    # If some companies already exist, include that information in the response
    if existing_companies:
        if successful_saves:
            message = f'Successfully saved {len(successful_saves)} new companies. {len(existing_companies)} companies already exist.'
        else:
            message = f'All {len(existing_companies)} companies already exist in Airtable.'
    else:
        message = f'Successfully saved {len(successful_saves)} out of {len(companies)} companies'
    
    return JsonResponse({
        'success': len(successful_saves) > 0,
        'message': message,
        'companies_found': companies,
        'existing_companies': existing_companies,
        'new_companies': new_companies,
        'save_results': save_results
    })


@csrf_exempt
@require_POST
def ocr_ai_image_upload(request):
    try:
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'error': 'No image uploaded.'}, status=400)

        # Save upload to a temporary file path for the existing pipeline
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            for chunk in image_file.chunks():
                tmp.write(chunk)
            temp_path = tmp.name

        try:
            ai_client = AIClient()
            # Optimize to avoid 502/timeouts for large images - use more aggressive optimization
            optimized_path, cleanup_path = optimize_image_for_ocr(temp_path, max_side=1000, jpeg_quality=75)
            extracted_text = ai_client.extract_text_from_image(optimized_path)
            if not extracted_text or extracted_text.startswith("Error") or len(extracted_text.strip()) < 40:
                # Use direct image summarization when OCR is weak
                summary_obj = ai_client.summarize_image(optimized_path)
                if hasattr(ai_client, 'generate_structured_reply'):
                    reply = ai_client.generate_structured_reply(summary_obj.get('transcript', '') or summary_obj.get('summary', ''))
                else:
                    resp = ai_client.generate_response(summary_obj.get('transcript', '') or summary_obj.get('summary', ''), is_followup=True)
                    reply = resp.get('content', '')
                return JsonResponse({'text': summary_obj.get('transcript', ''), 'summary': summary_obj.get('summary', ''), 'reply': reply})

            # Summarize/contextualize using the existing text -> response generator
            if hasattr(ai_client, 'generate_structured_reply'):
                reply = ai_client.generate_structured_reply(extracted_text)
            else:
                resp = ai_client.generate_response(extracted_text, is_followup=True)
                reply = resp.get('content', '')
            return JsonResponse({'text': extracted_text, 'summary': reply, 'reply': reply})
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass
            try:
                if cleanup_path and os.path.exists(cleanup_path):
                    os.unlink(cleanup_path)
            except Exception:
                pass
    except Exception as e:
        logger.exception("ocr_ai_image_upload failed")
        return JsonResponse({'error': str(e)}, status=500)


# Real-time AI Response System
import json
import time
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

@csrf_exempt
@require_POST
def stream_ai_response(request):
    """Stream AI response in real-time using Server-Sent Events"""
    try:
        data = json.loads(request.body)
        message_content = data.get('message', '').strip()
        message_id = data.get('message_id')
        conversation_history = data.get('conversation_history', [])
        
        if not message_content:
            return JsonResponse({'error': 'No message provided'}, status=400)
        
        def generate_stream():
            try:
                ai_client = AIClient()
                
                # Build conversation context
                messages = []
                
                # Add system message
                system_message = """You are a helpful AI assistant, like ChatGPT. Analyze the text carefully and respond naturally and conversationally.

CRITICAL: This appears to be content from a WhatsApp chat or messaging conversation. You MUST:
1. Read and understand the ACTUAL conversation content
2. Identify specific details mentioned (names, companies, banking info, etc.)
3. Acknowledge the specific information in your response
4. Respond as if you understand the context of the conversation

Rules:
- If you see banking details (IBAN, SWIFT codes, account numbers), mention them specifically
- If you see company names or business information, acknowledge them
- If you see names of people, reference them appropriately
- If it's about wire transfers or financial transactions, be specific about what you see
- Respond naturally, like you're having a conversation
- Be helpful, friendly, and engaging
- Don't use formal greetings (Hello/Hi) or signatures (Best regards)
- Don't include placeholders like [Your Name]
- Just respond naturally to what the person is actually discussing
- Keep it conversational and helpful
- Show that you understand the specific context and details
- Respond in the same language as the user's message"""
                messages.append({"role": "system", "content": system_message})
                
                # Add conversation history
                for msg in conversation_history:
                    if msg.get('role') and msg.get('content'):
                        messages.append({
                            "role": msg['role'],
                            "content": msg['content']
                        })
                
                # Add current user message
                messages.append({"role": "user", "content": message_content})
                
                # Send initial typing indicator
                yield f"data: {json.dumps({'type': 'typing', 'content': 'AI is thinking...'})}\n\n"
                
                # Generate streaming response
                if USING_OPENAI_V1:
                    response = ai_client.client.chat.completions.create(
                        model=ai_client.text_model,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1000,
                        stream=True
                    )
                    
                    full_response = ""
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"
                            time.sleep(0.05)  # Small delay for realistic typing effect
                else:
                    # Fallback for older OpenAI API
                    response_data = ai_client.generate_response(message_content, is_followup=True)
                    full_response = response_data['content']
                    
                    # Simulate streaming by sending chunks
                    words = full_response.split()
                    for i, word in enumerate(words):
                        chunk = word + (" " if i < len(words) - 1 else "")
                        yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
                        time.sleep(0.1)
                
                # Send completion signal
                yield f"data: {json.dumps({'type': 'complete', 'content': full_response})}\n\n"
                
                # Update message in database if message_id provided
                if message_id:
                    try:
                        message = Message.objects.get(id=message_id)
                        message.final_response = full_response
                        message.save()
                    except Message.DoesNotExist:
                        pass
                        
            except Exception as e:
                logger.error(f"Error in stream generation: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'content': f'Error: {str(e)}'})}\n\n"
        
        response = StreamingHttpResponse(
            generate_stream(),
            content_type='text/event-stream'
        )
        response['Cache-Control'] = 'no-cache'
        response['Connection'] = 'keep-alive'
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Headers'] = 'Cache-Control'
        
        return response
        
    except Exception as e:
        logger.error(f"Error in stream_ai_response: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
@require_POST
def real_time_chat(request):
    """Real-time chat endpoint for instant AI responses"""
    try:
        data = json.loads(request.body)
        message_content = data.get('message', '').strip()
        conversation_id = data.get('conversation_id')
        
        if not message_content:
            return JsonResponse({'error': 'No message provided'}, status=400)
        
        # Initialize AI client
        ai_client = AIClient()
        
        # Detect language
        detected_language = detect_language(message_content)
        
        # Generate response with language context and conversation history
        conversation_history = data.get('conversation_history', [])
        
        # Use the enhanced generate_structured_reply method for better context understanding
        logger.info("Forcing use of generate_structured_reply method")
        try:
            ai_response = ai_client.generate_structured_reply(message_content)
            logger.info("Successfully used generate_structured_reply")
        except Exception as e:
            logger.error(f"Error using generate_structured_reply: {e}")
            # Fallback to generate_response with language detection
            response_data = ai_client.generate_response(
                message_content, 
                is_followup=True, 
                conversation_history=conversation_history,
                detected_language=detected_language
            )
            ai_response = response_data['content']
        
        # Create or update conversation
        if conversation_id:
            try:
                message = Message.objects.get(id=conversation_id)
                # Update conversation history
                if not message.conversation_history:
                    message.conversation_history = []
                message.conversation_history.append({
                    'role': 'user',
                    'content': message_content,
                    'timestamp': timezone.now().isoformat()
                })
                message.conversation_history.append({
                    'role': 'assistant',
                    'content': ai_response,
                    'timestamp': timezone.now().isoformat()
                })
                message.final_response = ai_response
                message.save()
            except Message.DoesNotExist:
                pass
        
        return JsonResponse({
            'success': True,
            'response': ai_response,
            'language': detected_language,
            'conversation_id': conversation_id
        })
        
    except Exception as e:
        logger.error(f"Error in real_time_chat: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def detect_language(text):
    """Enhanced language detection based on common patterns and linguistic features"""
    if not text or not text.strip():
        return 'English'
    
    text_lower = text.lower().strip()
    
    # Italian indicators - more comprehensive
    italian_words = [
        'ciao', 'grazie', 'prego', 'buongiorno', 'buonasera', 'buonanotte',
        'come', 'stai', 'bene', 'male', 'perfetto', 'ottimo', 'eccellente',
        'scusi', 'mi dispiace', 'arrivederci', 'salve', 'piacere', 'conosci',
        'parlare', 'capire', 'aiutare', 'lavoro', 'casa', 'famiglia',
        'italiano', 'italia', 'roma', 'milano', 'napoli', 'firenze',
        'grazie mille', 'per la', 'per il', 'molto', 'molta', 'mille'
    ]
    
    # Spanish indicators - more comprehensive
    spanish_words = [
        'hola', 'gracias', 'por favor', 'buenos días', 'buenas tardes', 'buenas noches',
        'como', 'estas', 'bien', 'mal', 'perfecto', 'excelente', 'muy bien',
        'disculpe', 'lo siento', 'hasta luego', 'saludos', 'gusto', 'conocer',
        'hablar', 'entender', 'ayudar', 'trabajo', 'casa', 'familia',
        'español', 'españa', 'madrid', 'barcelona', 'mexico', 'argentina',
        'muchas gracias', 'por tu', 'por su', 'muy', 'más'
    ]
    
    # Portuguese indicators - more comprehensive
    portuguese_words = [
        'olá', 'obrigado', 'obrigada', 'por favor', 'bom dia', 'boa tarde', 'boa noite',
        'como', 'está', 'bem', 'mal', 'perfeito', 'excelente', 'muito bem',
        'desculpe', 'sinto muito', 'até logo', 'saudações', 'prazer', 'conhecer',
        'falar', 'entender', 'ajudar', 'trabalho', 'casa', 'família',
        'português', 'portugal', 'brasil', 'lisboa', 'porto', 'rio de janeiro'
    ]
    
    # Count matches for each language
    italian_count = sum(1 for word in italian_words if word in text_lower)
    spanish_count = sum(1 for word in spanish_words if word in text_lower)
    portuguese_count = sum(1 for word in portuguese_words if word in text_lower)
    
    # Check for specific linguistic patterns
    # Italian patterns
    if 'ch' in text_lower and ('che' in text_lower or 'chi' in text_lower):
        italian_count += 2
    if text_lower.endswith(('zione', 'sione', 'tore', 'tori')):
        italian_count += 2
    
    # Spanish patterns
    if 'ñ' in text_lower or 'll' in text_lower:
        spanish_count += 2
    if text_lower.endswith(('ción', 'sión', 'mente')):
        spanish_count += 2
    
    # Portuguese patterns
    if 'ã' in text_lower or 'õ' in text_lower or 'ç' in text_lower:
        portuguese_count += 2
    if text_lower.endswith(('ção', 'são', 'mente')):
        portuguese_count += 2
    
    # English indicators - add some common English words
    english_words = [
        'hello', 'hi', 'thank you', 'thanks', 'please', 'good morning', 'good afternoon',
        'how are you', 'fine', 'well', 'help', 'work', 'home', 'family', 'name',
        'english', 'england', 'america', 'london', 'new york', 'very much', 'for your'
    ]
    english_count = sum(1 for word in english_words if word in text_lower)
    
    # Return language with highest count, default to English
    if italian_count > spanish_count and italian_count > portuguese_count and italian_count > english_count:
        return 'Italian'
    elif spanish_count > portuguese_count and spanish_count > english_count:
        return 'Spanish'
    elif portuguese_count > english_count:
        return 'Portuguese'
    else:
        return 'English'


def real_time_chat_view(request):
    """View for the real-time chat interface"""
    return render(request, 'app/real_time_chat.html')


