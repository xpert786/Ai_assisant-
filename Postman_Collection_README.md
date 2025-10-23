# AI Project API Collection

This Postman collection provides a comprehensive set of API endpoints for testing the AI-powered Django application with OCR, real-time chat, and message management features.

## Collection Overview

The collection is organized into the following folders:

### 🔐 Authentication & Setup
- **Check API Status**: Verify Django admin accessibility

### 💬 Real-time AI Chat
- **Stream AI Response**: Stream AI responses in real-time using Server-Sent Events
- **Real-time Chat**: Instant AI chat responses
- **Chat History**: Get recent chat messages
- **Chat Detail**: Get detailed information about specific messages

### 📷 OCR & Image Processing
- **OCR AI Image Upload**: Upload images for OCR processing with AI analysis

### 📝 Message Management
- **Create Message**: Create new messages for AI processing
- **Get Message Detail**: Retrieve detailed message information
- **Get Message JSON**: Get message data as JSON (includes image URLs)
- **Update Message**: Modify AI responses or final responses
- **Delete Message**: Remove specific messages
- **Generate PDF**: Create PDFs from message responses
- **Enhance Response**: Improve responses with AI follow-ups

### 👥 Client Management
- **Search Clients**: Search across Clients, Accounts, and Leads tables
- **Create Lead**: Generate leads from messages with client information

### 🧠 Knowledge Base & Integration
- **Save to Knowledge Base**: Store responses in knowledge base
- **Save to Airtable Accounts**: Save data to Airtable Accounts
- **Approve and Save**: Approve and save documents/logs/templates

### 🏠 Dashboard & Views
- **Dashboard**: Main application dashboard
- **Real-time Chat View**: Chat interface view

## Setup Instructions

### 1. Environment Variables

Create a `.env` file in your project root or set the following variables in Postman:

```
OPENAI_API_KEY=your_openai_api_key_here
BASE_URL=http://localhost:8000/ai
```

### 2. Import the Collection

1. Open Postman
2. Click **Import** button
3. Select **Upload Files**
4. Choose `Ai_Project_Postman_Collection.json`
5. The collection will be imported with all requests

### 3. Configure Variables

1. In Postman, go to **Environments**
2. Create a new environment (e.g., "AI Project Dev")
3. Add variables:
   - `baseUrl`: `http://localhost:8000/ai`
   - `openai_api_key`: Your OpenAI API key

### 4. Authentication

Most endpoints don't require authentication, but some may need:
- Django session cookies for admin access
- OpenAI API key for AI features (set in environment variables)

## Usage Examples

### Real-time Chat
```bash
POST {{baseUrl}}/api/real-time-chat/
Content-Type: application/json

{
  "message": "Help me with business registration",
  "conversation_id": "chat_123"
}
```

### OCR Image Upload
```bash
POST {{baseUrl}}/api/ocr-ai-image-upload/
Content-Type: multipart/form-data

image: [select image file]
```

### Search Clients
```bash
POST {{baseUrl}}/api/search-clients/
Content-Type: application/json

{
  "query": "John Doe"
}
```

## Response Formats

### Success Response Example
```json
{
  "success": true,
  "response": "AI-generated response content...",
  "message_id": 123
}
```

### Error Response Example
```json
{
  "success": false,
  "error": "Error description",
  "message": "Detailed error message"
}
```

## Testing the APIs

1. **Start your Django server**: `python manage.py runserver`
2. **Import the collection** into Postman
3. **Set environment variables**
4. **Run requests** in the order they appear in folders
5. **Check test scripts** - each request includes basic validation tests

## Notes

- The collection assumes your Django server runs on `http://localhost:8000/ai`
- Some endpoints require file uploads (OCR) - use Postman's form-data option
- Real-time chat endpoints use Server-Sent Events for streaming responses
- The collection includes comprehensive test scripts for validation
- All requests are documented with expected request/response formats

## Troubleshooting

1. **Connection refused**: Ensure Django server is running
2. **Authentication errors**: Check if endpoints require login
3. **File upload issues**: Verify file paths and formats for OCR endpoints
4. **OpenAI errors**: Ensure valid API key is set in environment variables

## Collection Features

✅ **Comprehensive coverage** of all API endpoints
✅ **Proper request/response examples** with realistic data
✅ **Test scripts** for response validation
✅ **Environment variable support**
✅ **Organized folder structure**
✅ **Detailed descriptions** for each endpoint
✅ **Error handling examples**

