import os
from dotenv import load_dotenv
import tempfile
import traceback
import logging
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
import pymupdf4llm
import openai
import genanki
import random
import time
from youtube_transcript_api import YouTubeTranscriptApi
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
from youtube_transcript_api.proxies import WebshareProxyConfig
import asyncio
import requests
import json
import re
from urllib.parse import parse_qs, urlparse
from xml.etree import ElementTree as ET

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info("Starting MemoriAI")

# Initialise FastAPI app
app = FastAPI(title="MemoriAI Anki Deck Generator")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions globally."""
    error_msg = f"Unexpected error: {str(exc)}"
    print(f"Global exception: {error_msg}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": error_msg},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    error_msg = f"Validation error: {str(exc)}"
    print(f"Validation exception: {error_msg}")
    return JSONResponse(
        status_code=422,
        content={"detail": error_msg},
    )

# Configure DeepSeek API (using OpenAI's client)
# Replace with your DeepSeek API key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# Use OpenAI's client but with DeepSeek's base URL
openai.api_key = DEEPSEEK_API_KEY
openai.base_url = "https://api.deepseek.com"

if not DEEPSEEK_API_KEY:
    print("Warning: DEEPSEEK_API_KEY environment variable is not set")

WEBSHARE_USERNAME = os.getenv('WEBSHARE_USERNAME')
WEBSHARE_PASSWORD = os.getenv('WEBSHARE_PASSWORD')

# Validate credentials are present
if not WEBSHARE_USERNAME or not WEBSHARE_PASSWORD:
    print("⚠️  WARNING: Webshare credentials not found!")
    print("📝 Set these environment variables:")
    print("   - WEBSHARE_USERNAME")
    print("   - WEBSHARE_PASSWORD")
    print("")
    print("🖥️  For local development:")
    print("   export WEBSHARE_USERNAME='your_username'")
    print("   export WEBSHARE_PASSWORD='your_password'")
    print("")
    print("☁️  For Render deployment:")
    print("   Add them in the Environment section of your service settings")

# Models
class ProcessingStatus(BaseModel):
    task_id: str
    status: str
    progress: float = 0.0
    message: Optional[str] = None
    estimated_count: Optional[int] = None

class QAPair(BaseModel):
    question: str
    answer: str

class DeckRequest(BaseModel):
    title: str
    qa_pairs: List[QAPair]

class ProcessTextRequest(BaseModel):
    text: str
    num_pairs: int = 0  # 0 means use the estimated count

class WebshareFreeTierManager:
    """
    Manage Webshare free tier proxies with built-in rate limiting.
    """
    
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.proxy_config = None
        self.requests_made = 0
        self.bandwidth_used_mb = 0
        self.monthly_limit_mb = 1024  # 1GB = 1024MB
        self.last_request_time = 0
        self.min_delay_seconds = 2  # Minimum delay between requests
    
    def setup_proxy_config(self):
        """Set up the Webshare proxy configuration."""
        if not self.username or not self.password or self.username == 'your_webshare_username_here':
            raise ValueError("Please set your Webshare username and password!")
        
        self.proxy_config = WebshareProxyConfig(
            proxy_username=self.username,
            proxy_password=self.password
        )
        
        print(f"✅ Webshare proxy configured for user: {self.username}")
        return self.proxy_config
    
    async def wait_for_rate_limit(self):
        """Ensure we don't make requests too quickly."""
        now = time.time()
        time_since_last = now - self.last_request_time
        
        if time_since_last < self.min_delay_seconds:
            wait_time = self.min_delay_seconds - time_since_last
            print(f"⏱️  Rate limiting: waiting {wait_time:.1f} seconds...")
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def check_bandwidth_limit(self, estimated_request_size_mb: float = 0.5):
        """Check if we're approaching bandwidth limits."""
        if self.bandwidth_used_mb + estimated_request_size_mb > self.monthly_limit_mb:
            raise Exception(f"Approaching bandwidth limit! Used: {self.bandwidth_used_mb}MB / {self.monthly_limit_mb}MB")
        
        # Warn at 80% usage
        usage_percent = (self.bandwidth_used_mb / self.monthly_limit_mb) * 100
        if usage_percent > 80:
            print(f"⚠️  Warning: {usage_percent:.1f}% of monthly bandwidth used")
    
    def record_successful_request(self, estimated_size_mb: float = 0.5):
        """Record a successful request for tracking."""
        self.requests_made += 1
        self.bandwidth_used_mb += estimated_size_mb
        
        print(f"📊 Request #{self.requests_made} completed. Bandwidth used: {self.bandwidth_used_mb:.1f}MB / {self.monthly_limit_mb}MB")
    
    def get_usage_stats(self):
        """Get current usage statistics."""
        return {
            'requests_made': self.requests_made,
            'bandwidth_used_mb': self.bandwidth_used_mb,
            'bandwidth_remaining_mb': self.monthly_limit_mb - self.bandwidth_used_mb,
            'estimated_requests_remaining': int((self.monthly_limit_mb - self.bandwidth_used_mb) / 0.5)
        }

webshare_manager = WebshareFreeTierManager(WEBSHARE_USERNAME, WEBSHARE_PASSWORD)


async def fetch_transcript_with_webshare_free(video_id: str, max_retries: int = 3) -> str:
    """
    Fetch YouTube transcript using Webshare free tier proxies.
    """
    
    # Set up proxy config if not already done
    if not webshare_manager.proxy_config:
        webshare_manager.setup_proxy_config()
    
    for attempt in range(max_retries):
        try:
            # Check bandwidth limits
            webshare_manager.check_bandwidth_limit()
            
            # Rate limiting
            await webshare_manager.wait_for_rate_limit()
            
            print(f"🔄 Attempt {attempt + 1}: Fetching transcript for {video_id} via Webshare proxy")
            
            # Create YouTube API instance with Webshare proxy
            ytt_api = YouTubeTranscriptApi(proxy_config=webshare_manager.proxy_config)
            
            # Fetch transcript
            transcript_list = ytt_api.get_transcript(video_id)
            transcript_text = ""
            for item in transcript_list:
                transcript_text += item['text'] + " "
            
            # Record successful request
            webshare_manager.record_successful_request()
            
            print(f"✅ Success! Fetched {len(transcript_text)} characters via Webshare proxy")
            return transcript_text
            
        except Exception as e:
            error_msg = str(e).lower()
            
            print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
            
            # Check if it's a rate limiting error
            if "429" in error_msg or "too many requests" in error_msg:
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = (2 ** attempt) + random.uniform(1, 3)
                    print(f"⏳ Rate limited. Waiting {wait_time:.1f} seconds before retry...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise Exception("Rate limit exceeded after all retries with Webshare proxy")
            
            # Check if it's a credential/proxy error
            elif any(keyword in error_msg for keyword in ['auth', 'credential', 'proxy', 'connection']):
                raise Exception(f"Webshare proxy error: {str(e)}. Check your username/password.")
            
            # Other errors (video not available, etc.)
            elif "not available" in error_msg or "unavailable" in error_msg:
                raise Exception(f"Video unavailable: {str(e)}")
            
            # Generic error - try again
            elif attempt < max_retries - 1:
                wait_time = 2 + random.uniform(0.5, 1.5)
                print(f"⏳ Retrying in {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time)
                continue
            else:
                raise Exception(f"Failed after all attempts: {str(e)}")
    
    raise Exception("Failed to fetch transcript after all attempts")



# In-memory storage for task status
tasks_status = {}

# Cleanup old tasks and temporary files
@app.on_event("startup")
async def setup_periodic_cleanup():
    """Set up periodic cleanup of old tasks and temporary files."""
    import asyncio
    
    async def cleanup_old_tasks():
        """Clean up old tasks and their temporary files periodically."""
        while True:
            try:
                print("Running periodic cleanup of old tasks and temporary files...")
                current_time = time.time()
                tasks_to_remove = []
                
                # Find tasks older than 24 hours
                for task_id, task_data in tasks_status.items():
                    # Skip if the task doesn't have a timestamp
                    if not task_data.get("timestamp"):
                        task_data["timestamp"] = current_time
                        continue
                    
                    # Check if the task is older than 24 hours
                    if current_time - task_data["timestamp"] > 24 * 60 * 60:
                        tasks_to_remove.append(task_id)
                        
                        # Clean up associated files
                        if task_data.get("temp_dir") and os.path.exists(task_data["temp_dir"]):
                            try:
                                import shutil
                                shutil.rmtree(task_data["temp_dir"], ignore_errors=True)
                                print(f"Cleaned up temporary directory for task {task_id}: {task_data['temp_dir']}")
                            except Exception as e:
                                print(f"Error cleaning up files for task {task_id}: {str(e)}")
                
                # Remove old tasks from the dictionary
                for task_id in tasks_to_remove:
                    del tasks_status[task_id]
                    print(f"Removed old task: {task_id}")
                
                # Wait for 1 hour before next cleanup
                await asyncio.sleep(60 * 60)
            except Exception as e:
                print(f"Error in periodic cleanup: {str(e)}")
                # Wait for 10 minutes before retrying after an error
                await asyncio.sleep(10 * 60)
    
    # Start the cleanup task
    asyncio.create_task(cleanup_old_tasks())

# Add timestamp to task status
def add_timestamp_to_task(task_id: str):
    """Add a timestamp to a task for cleanup purposes."""
    if task_id in tasks_status:
        tasks_status[task_id]["timestamp"] = time.time()


@app.get("/", response_class=HTMLResponse)
async def get_html():
    return Path("ankiFrontEnd.html").read_text()
# Helper functions
@app.post("/upload/pdf", response_model=ProcessingStatus)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    num_pairs: Optional[int] = None
):
    try:
        # Generate a unique task ID
        task_id = f"task_{random.randint(10000, 99999)}"
        
        # Read the file content immediately
        logger.info(f"Reading file content for {file.filename}")
        file_content = await file.read()
        file_size = len(file_content)
        
        logger.info(f"Read {file_size} bytes from {file.filename}")
        
        # Check file size
        if file_size > 20 * 1024 * 1024:  # 20MB
            raise ValueError("PDF file is too large (over 20MB). Please upload a smaller file.")
        
        # Create a temporary file to count pages
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(file_content)
        temp_file.close()
        
        # Count pages in the PDF
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(temp_file.name)
            page_count = len(doc)
            doc.close()
        except Exception as e:
            logger.warning(f"Could not count PDF pages: {str(e)}")
            page_count = max(1, file_size // (100 * 1024))  # Rough estimate based on file size
        finally:
            os.unlink(temp_file.name)
        
        # Estimate appropriate number of cards
        estimated_count = estimate_card_count("pdf", page_count)
        
        # Use provided count if specified, otherwise use estimate
        if num_pairs is None or num_pairs <= 0:
            num_pairs = estimated_count
        else:
            num_pairs = min(50, max(1, int(num_pairs)))
        
        logger.info(f"Estimated {estimated_count} cards for {page_count} pages, using {num_pairs}")
        
        # Initialize task status
        tasks_status[task_id] = {
            "status": "processing",
            "progress": 0.0,
            "message": f"Initializing (generating {num_pairs} Q&A pairs)",
            "file_path": None,
            "qa_pairs": None,
            "estimated_count": estimated_count
        }
        
        # Add timestamp to track task age
        add_timestamp_to_task(task_id)
        
        # Start background processing with file content
        background_tasks.add_task(process_pdf, task_id, file_content, file.filename, num_pairs)
        
        return ProcessingStatus(
            task_id=task_id,
            status="processing",
            progress=0.0,
            message=f"PDF upload received, processing started (generating {num_pairs} Q&A pairs)",
            estimated_count=estimated_count
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error in upload_pdf: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")

async def process_pdf(task_id: str, file_content: bytes, filename: str, num_pairs: int):
    """Process the PDF file in the background."""
    temp_dir = None
    try:
        logger.info(f"Starting PDF processing for task {task_id}")
        logger.debug(f"File name: {filename}, Content size: {len(file_content)} bytes")
        
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, "input.pdf")
        logger.debug(f"Created temporary directory: {temp_dir}")
        logger.debug(f"Temporary file path: {file_path}")
        
        # Store the temp directory in task status immediately
        tasks_status[task_id].update({
            "temp_dir": temp_dir,
            "file_path": file_path,
            "progress": 0.1,
            "message": "Writing PDF file"
        })
        logger.debug(f"Updated task status: {tasks_status[task_id]}")
        
        # Write content to temporary file
        logger.debug(f"Writing content to temporary file: {file_path}")
        with open(file_path, "wb") as temp_file:
            temp_file.write(file_content)
        logger.info(f"Successfully wrote {len(file_content)} bytes to {file_path}")
        
        # Update status after successful file write
        tasks_status[task_id].update({
            "progress": 0.3,
            "message": "Extracting text from PDF"
        })
        
        # Extract text using pymupdf4llm
        try:
            logger.info("Extracting text using pymupdf4llm")
            text = pymupdf4llm.to_markdown(file_path)
            logger.info(f"Successfully extracted {len(text)} characters of text")
            logger.debug(f"First 200 characters of extracted text: {text[:200]}...")
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}", exc_info=True)
            tasks_status[task_id].update({
                "status": "failed",
                "message": f"Could not extract text from the PDF: {str(e)}"
            })
            return
        
        # Check if we got enough text
        if not text or len(text.strip()) < 50:
            logger.warning(f"Extracted text too short: {len(text.strip() if text else 0)} characters")
            tasks_status[task_id].update({
                "status": "failed",
                "message": "Could not extract enough text from the PDF. The file might be scanned images or protected."
            })
            return
        
        # Now process the text exactly like direct text entry
        # Update status for QA pair generation
        tasks_status[task_id].update({
            "progress": 0.5,
            "message": "Generating QA pairs"
        })
        
        # Process the text just like in process_text_task
        try:
            logger.info(f"Generating QA pairs from {len(text)} characters of text")
            # Generate QA pairs
            qa_pairs = generate_qa_pairs(text, num_pairs)
            logger.info(f"Successfully generated {len(qa_pairs)} QA pairs")
            
            # Update status
            tasks_status[task_id].update({
                "progress": 1.0,
                "status": "completed",
                "message": "Processing complete",
                "qa_pairs": qa_pairs
            })
            logger.info(f"PDF processing completed successfully for task {task_id}")
        except Exception as e:
            logger.error(f"Error generating QA pairs: {str(e)}", exc_info=True)
            tasks_status[task_id].update({
                "status": "failed",
                "message": f"Error generating QA pairs: {str(e)}"
            })
            
    except Exception as e:
        logger.error(f"Unexpected error in process_pdf: {str(e)}", exc_info=True)
        tasks_status[task_id].update({
            "status": "failed",
            "message": f"Error processing PDF: {str(e)}"
        })
    finally:
        # Clean up temporary directory if it exists
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                logger.debug(f"Cleaning up temporary directory: {temp_dir}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.debug("Temporary directory cleanup complete")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {str(e)}")

def generate_qa_pairs(text: str, num_pairs: int) -> List[Dict[str, str]]:
    """Generate question-answer pairs using DeepSeek API."""
    try:
        # Check if API key is set
        if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your-deepseek-api-key-here":
            logger.error("DeepSeek API key is not set")
            return [{"question": "Error: DeepSeek API key is not set", 
                    "answer": "Please set your DeepSeek API key in the main.py file."}]
        
        # Ensure text is not empty
        if not text or len(text.strip()) < 10:
            logger.warning("Input text is too short or empty")
            return [{"question": "Error: Input text is too short", 
                    "answer": "Please provide more text to generate meaningful questions."}]
        
        # Ensure num_pairs is a valid number
        try:
            num_pairs = int(num_pairs)
            if num_pairs < 1:
                num_pairs = 1
            elif num_pairs > 50:
                num_pairs = 50
            logger.info(f"Generating {num_pairs} Q&A pairs")
        except (ValueError, TypeError):
            logger.warning(f"Invalid num_pairs value: {num_pairs}")
        
        logger.info(f"Sending request to DeepSeek API with {len(text[:4000])} characters of text for {num_pairs} Q&A pairs")
        
        # Using OpenAI client with DeepSeek API
        client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        
        # Truncate text if it's too long (DeepSeek has token limits)
        max_chars = 12000  # Approximate character limit
        if len(text) > max_chars:
            logger.warning(f"Text too long ({len(text)} chars), truncating to {max_chars} chars")
            text = text[:max_chars] + "..."
        
        # Create the prompt with explicit number of pairs
        prompt = f"""
        Based on the following text, generate EXACTLY {num_pairs} question-answer pairs for flashcards.
        You MUST generate EXACTLY {num_pairs} pairs, no more and no less.
        Each question should test understanding of a key concept, and the answer should be comprehensive.
        
        TEXT:
        {text}
        
        FORMAT:
        Return the output as a JSON array of objects, each with 'question' and 'answer' fields.
        Do not include any explanations or other text outside the JSON structure.
        The array MUST contain EXACTLY {num_pairs} items.
        """
        
        logger.debug(f"Requesting {num_pairs} Q&A pairs from DeepSeek API")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant that generates high-quality question-answer pairs for flashcards. You will generate EXACTLY {num_pairs} pairs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        
        logger.debug("Received response from DeepSeek API")
        content = response.choices[0].message.content
        logger.debug(f"Response content: {content[:200]}...")
        
        # Parse the response
        import json
        import re
        
        # Try to extract JSON from the response
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
        if json_match:
            logger.debug("Found JSON in code block")
            json_str = json_match.group(1)
        else:
            logger.debug("No JSON code block found, using entire response")
            json_str = content
        
        # Clean up the string to ensure it's valid JSON
        json_str = re.sub(r'```.*', '', json_str)
        json_str = re.sub(r'^.*?\[', '[', json_str.strip(), count=1, flags=re.DOTALL)
        if not json_str.startswith('['):
            json_str = '[' + json_str
        if not json_str.endswith(']'):
            json_str = json_str + ']'
        
        logger.debug(f"Cleaned JSON string: {json_str[:200]}...")
        
        try:
            qa_pairs = json.loads(json_str)
            logger.info(f"Successfully parsed {len(qa_pairs)} QA pairs from response")
            
            # Validate the structure
            valid_pairs = []
            for pair in qa_pairs:
                if isinstance(pair, dict) and 'question' in pair and 'answer' in pair:
                    valid_pairs.append({
                        'question': pair['question'],
                        'answer': pair['answer']
                    })
            
            logger.info(f"Validated {len(valid_pairs)} QA pairs")
            
            # If we got more pairs than requested, trim the list
            if len(valid_pairs) > num_pairs:
                logger.warning(f"Got {len(valid_pairs)} pairs but only requested {num_pairs}, trimming list")
                valid_pairs = valid_pairs[:num_pairs]
            
            # If we got fewer pairs than requested, log a warning
            if len(valid_pairs) < num_pairs:
                logger.warning(f"Requested {num_pairs} pairs but only got {len(valid_pairs)}")
            
            return valid_pairs
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            logger.error(f"Problematic JSON string: {json_str}")
            return [{"question": "Error parsing API response", 
                    "answer": "The API returned a response that couldn't be parsed as JSON."}]
            
    except Exception as e:
        logger.error(f"Error in generate_qa_pairs: {str(e)}", exc_info=True)
        return [{"question": "Error generating QA pairs", 
                "answer": f"An error occurred: {str(e)}"}]

def create_anki_deck(title: str, qa_pairs: List[Dict[str, str]]) -> str:
    """Create an Anki deck from QA pairs and return the file path."""
    # Create a unique model ID and deck ID
    model_id = random.randrange(1 << 30, 1 << 31)
    deck_id = random.randrange(1 << 30, 1 << 31)

    logger.info(f"Creating Anki deck with title: {title}")
    
    # Define the card model (template)
    model = genanki.Model(
        model_id,
        'Simple Model',
        fields=[
            {'name': 'Question'},
            {'name': 'Answer'},
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': '{{Question}}',
                'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}',
            },
        ])
    
    # Create the deck
    deck = genanki.Deck(deck_id, title)
    
    # Add notes (cards) to the deck
    for qa in qa_pairs:
        note = genanki.Note(
            model=model,
            fields=[qa['question'], qa['answer']]
        )
        deck.add_note(note)
    
    # Create a temporary file to store the deck
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, f"{title.replace(' ', '_')}.apkg")
    
    # Save the deck to the file
    genanki.Package(deck).write_to_file(output_path)
    
    return output_path

@app.get("/status/{task_id}", response_model=ProcessingStatus)
async def get_status(task_id: str):
    """Get the status of a processing task."""
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    status = tasks_status[task_id]
    return ProcessingStatus(
        task_id=task_id,
        status=status["status"],
        progress=status["progress"],
        message=status["message"]
    )

@app.post("/generate-deck", response_model=ProcessingStatus)
async def generate_deck(request: DeckRequest, background_tasks: BackgroundTasks):
    """Generate an Anki deck from provided QA pairs."""
    # Generate a unique task ID
    task_id = f"deck_{random.randint(10000, 99999)}"
    
    # Initialise task status
    tasks_status[task_id] = {
        "status": "processing",
        "progress": 0.0,
        "message": "Starting deck generation",
        "file_path": None,
        "timestamp": time.time()  # Add timestamp
    }
    
    # Generate the deck in the background
    background_tasks.add_task(create_deck_task, task_id, request.title, request.qa_pairs)
    
    return ProcessingStatus(task_id=task_id, status="processing", progress=0.0)

async def create_deck_task(task_id: str, title: str, qa_pairs: List[QAPair]):
    """Create an Anki deck in the background."""
    temp_dir = None
    try:
        # Convert QA pairs to the format expected by create_anki_deck
        qa_dict_pairs = [{"question": qa.question, "answer": qa.answer} for qa in qa_pairs]
        
        # Update status
        tasks_status[task_id]["progress"] = 0.3
        tasks_status[task_id]["message"] = "Creating Anki deck"
        
        # Create the deck
        file_path = create_anki_deck(title, qa_dict_pairs)
        temp_dir = os.path.dirname(file_path)
        
        # Update status
        tasks_status[task_id]["progress"] = 1.0
        tasks_status[task_id]["status"] = "completed"
        tasks_status[task_id]["message"] = "Deck generation complete"
        tasks_status[task_id]["file_path"] = file_path
        
    except Exception as e:
        tasks_status[task_id]["status"] = "failed"
        tasks_status[task_id]["message"] = f"Error: {str(e)}"
        
        # Clean up if error occurs and we have a temp directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as cleanup_error:
                print(f"Warning: Failed to clean up temporary files after error: {cleanup_error}")

@app.get("/download/{task_id}")
async def download_deck(task_id: str, background_tasks: BackgroundTasks):
    """Download the generated Anki deck."""
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    status = tasks_status[task_id]
    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Deck generation not completed")
    
    if not status.get("file_path") or not os.path.exists(status["file_path"]):
        raise HTTPException(status_code=404, detail="Deck file not found")
    
    file_path = status["file_path"]
    filename = os.path.basename(file_path)
    
    # Schedule cleanup to run after response is sent
    # This ensures the file is available for download but cleaned up afterward
    def cleanup_temp_files():
        try:
            # Wait a bit to ensure the file has been sent
            import time
            time.sleep(60)  # Wait 60 seconds before cleanup
            
            # Get the directory containing the file
            if status.get("temp_dir") and os.path.exists(status["temp_dir"]):
                import shutil
                shutil.rmtree(status["temp_dir"], ignore_errors=True)
                print(f"Cleaned up temporary directory after download: {status['temp_dir']}")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
    
    background_tasks.add_task(cleanup_temp_files)
    
    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=filename
    )

@app.get("/qa-pairs/{task_id}")
async def get_qa_pairs(task_id: str):
    """Get the generated QA pairs for a task."""
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    status = tasks_status[task_id]
    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed")
    
    if not status.get("qa_pairs"):
        raise HTTPException(status_code=404, detail="QA pairs not found")
    
    return {"qa_pairs": status["qa_pairs"]}

@app.post("/process/text", response_model=ProcessingStatus)
async def process_text(request: ProcessTextRequest, background_tasks: BackgroundTasks):
    # Generate a unique task ID
    task_id = f"task_{random.randint(10000, 99999)}"
    
    # Estimate appropriate number of cards based on text length
    text_length = len(request.text)
    estimated_count = estimate_card_count("text", text_length)
    
    # Use provided count if specified, otherwise use estimate
    num_pairs = request.num_pairs if request.num_pairs > 0 else estimated_count
    num_pairs = min(50, max(1, num_pairs))
    
    logger.info(f"Estimated {estimated_count} cards for {text_length} chars, using {num_pairs}")
    
    # Initialize task status
    tasks_status[task_id] = {
        "status": "processing",
        "progress": 0.0,
        "message": "Starting text processing",
        "qa_pairs": None,
        "timestamp": time.time(),
        "estimated_count": estimated_count
    }
    
    # Process the text in the background
    background_tasks.add_task(process_text_task, task_id, request.text, num_pairs)
    
    return ProcessingStatus(
        task_id=task_id, 
        status="processing", 
        progress=0.0,
        estimated_count=estimated_count
    )

async def process_text_task(task_id: str, text: str, num_pairs: int):
    """Process text in the background."""
    try:
        # Validate num_pairs
        try:
            num_pairs = int(num_pairs)
            if num_pairs < 1:
                num_pairs = 1
            elif num_pairs > 50:
                num_pairs = 50
            logger.info(f"Processing text to generate {num_pairs} Q&A pairs")
        except (ValueError, TypeError):
            logger.warning(f"Invalid num_pairs value: {num_pairs}, using default of 10")
            num_pairs = 10
            
        # Update status
        tasks_status[task_id].update({
            "progress": 0.3,
            "message": f"Processing text (generating {num_pairs} Q&A pairs)"
        })
        
        # Check if text is too short
        if len(text.strip()) < 100:
            tasks_status[task_id].update({
                "status": "failed",
                "message": "The provided text is too short. Please provide more content to generate meaningful questions."
            })
            return
        
        # Generate QA pairs
        logger.info(f"Generating {num_pairs} Q&A pairs from {len(text)} characters of text")
        qa_pairs = generate_qa_pairs(text, num_pairs)
        logger.info(f"Successfully generated {len(qa_pairs)} Q&A pairs")
        
        # Update status
        tasks_status[task_id].update({
            "progress": 1.0,
            "status": "completed",
            "message": f"Processing complete ({len(qa_pairs)} Q&A pairs generated)",
            "qa_pairs": qa_pairs
        })
        
    except Exception as e:
        logger.error(f"Error in process_text_task: {str(e)}", exc_info=True)
        tasks_status[task_id].update({
            "status": "failed",
            "message": f"Error processing text: {str(e)}"
        })

@app.get("/test-deepseek")
async def test_deepseek():
    """Test if the DeepSeek API is working correctly."""
    try:
        if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your-deepseek-api-key-here":
            return {"status": "error", "message": "DeepSeek API key is not set. Please update the DEEPSEEK_API_KEY in main.py."}
        
        # Make a simple API call using the OpenAI client with DeepSeek API
        client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'DeepSeek API is working!' in JSON format with a status field."}
            ],
            temperature=0.7,
        )
        
        content = response.choices[0].message.content
        return {"status": "success", "message": "DeepSeek API is working correctly", "response": content}
    except Exception as e:
        return {"status": "error", "message": f"Error testing DeepSeek API: {str(e)}"}
    

def estimate_card_count(content_type, content_size):
    """
    Estimate appropriate number of flashcards based on content type and size.
    
    Parameters:
    - content_type: "pdf", "text", or "youtube"
    - content_size: For PDF: number of pages, for text: word count, for YouTube: duration in seconds
    
    Returns:
    - estimated_count: Recommended number of cards
    """
    if content_type == "pdf":
        # Estimate 2-3 cards per page
        estimated_count = max(5, round(content_size * 2.5))
    elif content_type == "text":
        # Estimate 1 card per 150 words
        word_count = content_size / 6  # Rough estimate of words from character count
        estimated_count = max(5, round(word_count / 150))
    elif content_type == "youtube":
        # Estimate 1 card per minute of content
        minutes = content_size / 60
        estimated_count = max(5, round(minutes))
    else:
        # Default fallback
        estimated_count = 10
    
    # Round to nearest 5 for cleaner presentation
    estimated_count = round(estimated_count / 5) * 5
    
    # Cap at reasonable maximum
    return min(estimated_count, 50)

def extract_youtube_id(url: str) -> str:
    """Extract YouTube video ID from URL."""
    import re
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Invalid YouTube URL format")

class YoutubeRequest(BaseModel):
    url: str
    num_pairs: int = 0  # 0 means use the estimated count

@app.post("/process/youtube", response_model=ProcessingStatus)
async def process_youtube_with_enhanced_webshare_integration(request: YoutubeRequest, background_tasks: BackgroundTasks):
    """
    Enhanced YouTube processing endpoint with improved extraction methods.
    """
    # Generate a unique task ID
    task_id = f"task_{random.randint(10000, 99999)}"
    
    # Estimate count
    estimated_duration_minutes = 10
    estimated_count = estimate_card_count("youtube", estimated_duration_minutes * 60)
    
    # Use provided count if specified, otherwise use estimate
    num_pairs = request.num_pairs if request.num_pairs > 0 else estimated_count
    num_pairs = min(50, max(1, num_pairs))
    
    # Initialize task status
    tasks_status[task_id] = {
        "status": "processing",
        "progress": 0.0,
        "message": "Starting enhanced YouTube transcript processing",
        "qa_pairs": None,
        "timestamp": time.time(),
        "estimated_count": estimated_count
    }
    
    # Process with enhanced methods in the background
    background_tasks.add_task(process_youtube_task_with_enhanced_webshare, task_id, request.url, num_pairs)
    
    return ProcessingStatus(
        task_id=task_id, 
        status="processing", 
        progress=0.0,
        message="Processing with enhanced extraction methods",
        estimated_count=estimated_count
    )

async def process_youtube_task_with_webshare(task_id: str, url: str, num_pairs: int):
    """
    Enhanced YouTube processing with Webshare free tier proxies.
    This replaces your existing process_youtube_task function.
    """
    try:
        # Validate num_pairs
        try:
            num_pairs = int(num_pairs)
            if num_pairs < 1:
                num_pairs = 1
            elif num_pairs > 50:
                num_pairs = 50
            logger.info(f"Processing YouTube transcript to generate {num_pairs} Q&A pairs")
        except (ValueError, TypeError):
            logger.warning(f"Invalid num_pairs value: {num_pairs}, using default of 10")
            num_pairs = 10
        
        # Update status
        tasks_status[task_id].update({
            "progress": 0.1,
            "message": "Extracting YouTube video ID"
        })
        
        # Extract video ID from URL
        try:
            video_id = extract_youtube_id(url)
            logger.info(f"Extracted YouTube video ID: {video_id}")
        except ValueError as e:
            tasks_status[task_id].update({
                "status": "failed", 
                "message": str(e)
            })
            return
        
        # Update status
        tasks_status[task_id].update({
            "progress": 0.3,
            "message": "Fetching transcript via Webshare proxy (free tier)"
        })
        
        # Fetch transcript with Webshare free tier
        try:
            transcript_text = await fetch_transcript_with_webshare_free(video_id)
            
            logger.info(f"Successfully fetched transcript: {len(transcript_text)} characters")
            
            # Check if transcript is too short
            if len(transcript_text.strip()) < 100:
                tasks_status[task_id].update({
                    "status": "failed",
                    "message": "The transcript is too short. Please try a different video with more content."
                })
                return
                
        except Exception as e:
            logger.error(f"Error fetching YouTube transcript with Webshare: {str(e)}")
            
            # Provide helpful error messages
            error_msg = str(e)
            if "username" in error_msg.lower() or "password" in error_msg.lower():
                tasks_status[task_id].update({
                    "status": "failed",
                    "message": "Webshare credentials error. Please check your username and password in the code."
                })
            elif "bandwidth" in error_msg.lower():
                tasks_status[task_id].update({
                    "status": "failed", 
                    "message": "Monthly bandwidth limit reached. Try again next month or upgrade to paid plan."
                })
            else:
                tasks_status[task_id].update({
                    "status": "failed",
                    "message": f"Failed to fetch transcript: {str(e)}"
                })
            return
        
        # Update status for QA generation
        tasks_status[task_id].update({
            "progress": 0.5,
            "message": f"Generating {num_pairs} Q&A pairs"
        })
        
        # Generate QA pairs from transcript
        qa_pairs = generate_qa_pairs(transcript_text, num_pairs)
        logger.info(f"Successfully generated {len(qa_pairs)} Q&A pairs")
        
        # Get usage stats for user info
        usage_stats = webshare_manager.get_usage_stats()
        
        # Update status
        tasks_status[task_id].update({
            "progress": 1.0,
            "status": "completed",
            "message": f"Processing complete ({len(qa_pairs)} Q&A pairs generated)",
            "qa_pairs": qa_pairs,
            "webshare_usage": usage_stats
        })
        
    except Exception as e:
        logger.error(f"Error in Webshare YouTube processing: {str(e)}", exc_info=True)
        tasks_status[task_id].update({
            "status": "failed",
            "message": f"Error processing YouTube transcript: {str(e)}"
        })

class EnhancedYouTubeExtractor:
    """
    Enhanced YouTube transcript extractor that combines multiple methods
    for better reliability with your existing WebShare setup.
    """
    
    def __init__(self, webshare_manager):
        self.webshare_manager = webshare_manager
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    async def extract_video_info(self, video_id: str) -> Dict[str, str]:
        """Get video title and basic info using oEmbed API."""
        try:
            oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            
            # Make request through WebShare proxy
            response = await self._make_proxy_request(oembed_url)
            
            if response and response.status_code == 200:
                data = response.json()
                return {
                    'title': data.get('title', f'YouTube Video ({video_id})'),
                    'author': data.get('author_name', 'Unknown'),
                    'duration': data.get('duration', 'Unknown')
                }
        except Exception as e:
            logger.warning(f"Failed to get video info: {e}")
        
        return {'title': f'YouTube Video ({video_id})', 'author': 'Unknown', 'duration': 'Unknown'}
    
    async def _make_proxy_request(self, url: str, max_retries: int = 3):
        """Make HTTP request through WebShare proxy with retry logic."""
        for attempt in range(max_retries):
            try:
                # Use existing WebShare proxy setup
                proxy_config = self.webshare_manager.setup_proxy_config()
                
                # Convert proxy config to requests format
                proxy_dict = {
                    'http': f'http://{proxy_config.proxy_username}:{proxy_config.proxy_password}@rotating-residential-proxies.all.webshare.io:8080',
                    'https': f'http://{proxy_config.proxy_username}:{proxy_config.proxy_password}@rotating-residential-proxies.all.webshare.io:8080'
                }
                
                response = requests.get(
                    url,
                    headers=self.headers,
                    proxies=proxy_dict,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                    
            except Exception as e:
                logger.warning(f"Proxy request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
        
        return None
    
    async def extract_transcript_from_watch_page(self, video_id: str) -> Optional[str]:
        """Extract transcript by parsing YouTube watch page through proxy."""
        watch_url = f"https://www.youtube.com/watch?v={video_id}"
        response = await self._make_proxy_request(watch_url)
        
        if not response:
            return None
        
        try:
            html_content = response.text
            
            # Look for ytInitialPlayerResponse in the HTML
            patterns = [
                r'var ytInitialPlayerResponse = ({.*?});',
                r'"captions":\s*({[^}]+})',
                r'"captionTracks":\s*\[([^\]]+)\]'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, html_content)
                if match:
                    try:
                        if pattern.startswith(r'var ytInitialPlayerResponse'):
                            player_data = json.loads(match.group(1))
                            transcript = await self._extract_from_player_response(player_data)
                            if transcript:
                                return transcript
                    except json.JSONDecodeError:
                        continue
            
            # Fallback: look for direct caption track URLs
            caption_url_pattern = r'"baseUrl":"([^"]*timedtext[^"]*)"'
            urls = re.findall(caption_url_pattern, html_content)
            
            for url in urls:
                if 'timedtext' in url:
                    transcript = await self._fetch_transcript_from_url(url.replace('\\u0026', '&'))
                    if transcript:
                        return transcript
                        
        except Exception as e:
            logger.error(f"Error parsing watch page: {e}")
        
        return None
    
    async def _extract_from_player_response(self, player_data: Dict) -> Optional[str]:
        """Extract transcript from ytInitialPlayerResponse data."""
        try:
            captions = player_data.get('captions', {})
            caption_tracks = captions.get('playerCaptionsTracklistRenderer', {}).get('captionTracks', [])
            
            if not caption_tracks:
                return None
            
            # Prefer English captions
            selected_track = None
            for track in caption_tracks:
                if track.get('languageCode', '').startswith('en'):
                    selected_track = track
                    break
            
            if not selected_track and caption_tracks:
                selected_track = caption_tracks[0]
            
            if selected_track:
                base_url = selected_track.get('baseUrl')
                if base_url:
                    return await self._fetch_transcript_from_url(base_url)
                    
        except Exception as e:
            logger.error(f"Error extracting from player response: {e}")
        
        return None
    
    async def _fetch_transcript_from_url(self, url: str) -> Optional[str]:
        """Fetch and parse transcript from timedtext URL."""
        if not url.startswith('http'):
            return None
        
        # Ensure we get the full transcript
        if '&fmt=' not in url:
            url += '&fmt=srv3'
        
        response = await self._make_proxy_request(url)
        if not response:
            return None
        
        try:
            # Parse XML response
            root = ET.fromstring(response.text)
            transcript_parts = []
            
            # Handle different XML formats
            for elem in root.iter():
                if elem.tag in ['text', 'p'] and elem.text:
                    text = elem.text.strip()
                    # Clean up the text
                    text = re.sub(r'\[.*?\]', '', text)  # Remove [Music], [Applause], etc.
                    text = re.sub(r'\s+', ' ', text)
                    if text and len(text) > 2:
                        transcript_parts.append(text)
            
            if transcript_parts:
                full_transcript = ' '.join(transcript_parts)
                # Final cleanup
                full_transcript = re.sub(r'\[Music\]|\[Applause\]|\[Laughter\]', '', full_transcript)
                full_transcript = re.sub(r'\s+', ' ', full_transcript).strip()
                
                return full_transcript if len(full_transcript) > 50 else None
                
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
        except Exception as e:
            logger.error(f"Error fetching transcript: {e}")
        
        return None

# Enhanced transcript extraction function
async def fetch_transcript_with_enhanced_webshare(video_id: str, max_retries: int = 3) -> str:
    """
    Enhanced transcript fetching that combines youtube-transcript-api 
    with direct extraction methods for better reliability.
    """
    
    # Initialize enhanced extractor
    enhanced_extractor = EnhancedYouTubeExtractor(webshare_manager)
    
    # Method 1: Try your existing youtube-transcript-api approach first
    try:
        logger.info(f"Trying Method 1: youtube-transcript-api for {video_id}")
        result = await fetch_transcript_with_webshare_free(video_id, max_retries)
        if result and len(result.strip()) > 100:
            logger.info("Method 1 successful")
            return result
    except Exception as e:
        logger.warning(f"Method 1 failed: {e}")
    
    # Method 2: Try enhanced direct extraction
    try:
        logger.info(f"Trying Method 2: Enhanced direct extraction for {video_id}")
        await webshare_manager.wait_for_rate_limit()
        
        transcript = await enhanced_extractor.extract_transcript_from_watch_page(video_id)
        if transcript and len(transcript.strip()) > 100:
            webshare_manager.record_successful_request()
            logger.info("Method 2 successful")
            return transcript
    except Exception as e:
        logger.warning(f"Method 2 failed: {e}")
    
    # Method 3: Fallback with different video URL formats
    try:
        logger.info(f"Trying Method 3: Alternative URL formats for {video_id}")
        
        # Try different URL formats that might work better
        alternative_urls = [
            f"https://www.youtube.com/embed/{video_id}",
            f"https://youtu.be/{video_id}",
            f"https://m.youtube.com/watch?v={video_id}"
        ]
        
        for url in alternative_urls:
            try:
                await webshare_manager.wait_for_rate_limit()
                response = await enhanced_extractor._make_proxy_request(url)
                if response:
                    # Try to extract any transcript information from the page
                    html_content = response.text
                    if 'captions' in html_content.lower() or 'transcript' in html_content.lower():
                        # If we find caption references, try the youtube-transcript-api again
                        ytt_api = YouTubeTranscriptApi(proxy_config=webshare_manager.proxy_config)
                        transcript_list = ytt_api.get_transcript(video_id)
                        transcript_text = " ".join([item['text'] for item in transcript_list])
                        
                        if len(transcript_text.strip()) > 100:
                            webshare_manager.record_successful_request()
                            logger.info("Method 3 successful")
                            return transcript_text
            except Exception as e:
                logger.warning(f"Alternative URL {url} failed: {e}")
                continue
                
    except Exception as e:
        logger.warning(f"Method 3 failed: {e}")
    
    # If all methods fail
    raise Exception("Could not extract transcript using any method. The video may not have captions available.")

# New endpoint for getting video info
@app.get("/youtube/info/{video_id}")
async def get_youtube_video_info(video_id: str):
    """Get YouTube video information including title and availability."""
    try:
        enhanced_extractor = EnhancedYouTubeExtractor(webshare_manager)
        video_info = await enhanced_extractor.extract_video_info(video_id)
        
        return {
            "status": "success",
            "video_id": video_id,
            "info": video_info
        }
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        return {
            "status": "error",
            "video_id": video_id,
            "error": str(e)
        }
    

async def process_youtube_task_with_enhanced_webshare(task_id: str, url: str, num_pairs: int):
    """
    Enhanced YouTube processing with multiple extraction methods.
    This replaces your existing process_youtube_task_with_webshare function.
    """
    try:
        # Validate num_pairs
        num_pairs = max(1, min(50, int(num_pairs)))
        
        # Update status
        tasks_status[task_id].update({
            "progress": 0.1,
            "message": "Extracting YouTube video ID"
        })
        
        # Extract video ID from URL
        try:
            video_id = extract_youtube_id(url)
            logger.info(f"Extracted YouTube video ID: {video_id}")
        except ValueError as e:
            tasks_status[task_id].update({
                "status": "failed", 
                "message": str(e)
            })
            return
        
        # Get video info
        tasks_status[task_id].update({
            "progress": 0.2,
            "message": "Getting video information"
        })
        
        enhanced_extractor = EnhancedYouTubeExtractor(webshare_manager)
        video_info = await enhanced_extractor.extract_video_info(video_id)
        
        # Update status for transcript extraction
        tasks_status[task_id].update({
            "progress": 0.3,
            "message": f"Fetching transcript for: {video_info['title']}"
        })
        
        # Fetch transcript with enhanced methods
        try:
            transcript_text = await fetch_transcript_with_enhanced_webshare(video_id)
            
            logger.info(f"Successfully fetched transcript: {len(transcript_text)} characters")
            
            # Check if transcript is too short
            if len(transcript_text.strip()) < 100:
                tasks_status[task_id].update({
                    "status": "failed",
                    "message": "The transcript is too short. Please try a different video with more content."
                })
                return
                
        except Exception as e:
            logger.error(f"Error fetching YouTube transcript: {str(e)}")
            
            # Provide helpful error messages
            error_msg = str(e)
            if "username" in error_msg.lower() or "password" in error_msg.lower():
                tasks_status[task_id].update({
                    "status": "failed",
                    "message": "Webshare credentials error. Please check your username and password."
                })
            elif "bandwidth" in error_msg.lower():
                tasks_status[task_id].update({
                    "status": "failed", 
                    "message": "Monthly bandwidth limit reached. Try again next month or upgrade to paid plan."
                })
            elif "not have captions" in error_msg.lower():
                tasks_status[task_id].update({
                    "status": "failed",
                    "message": "This video doesn't have captions or transcripts available. Please try a different video."
                })
            else:
                tasks_status[task_id].update({
                    "status": "failed",
                    "message": f"Failed to fetch transcript: {str(e)}"
                })
            return
        
        # Update status for QA generation
        tasks_status[task_id].update({
            "progress": 0.6,
            "message": f"Generating {num_pairs} Q&A pairs from transcript"
        })
        
        # Generate QA pairs from transcript
        qa_pairs = generate_qa_pairs(transcript_text, num_pairs)
        logger.info(f"Successfully generated {len(qa_pairs)} Q&A pairs")
        
        # Get usage stats
        usage_stats = webshare_manager.get_usage_stats()
        
        # Update status
        tasks_status[task_id].update({
            "progress": 1.0,
            "status": "completed",
            "message": f"Processing complete ({len(qa_pairs)} Q&A pairs generated)",
            "qa_pairs": qa_pairs,
            "video_info": video_info,
            "webshare_usage": usage_stats
        })
        
    except Exception as e:
        logger.error(f"Error in enhanced YouTube processing: {str(e)}", exc_info=True)
        tasks_status[task_id].update({
            "status": "failed",
            "message": f"Error processing YouTube transcript: {str(e)}"
        })


# Health check endpoint for YouTube functionality
@app.get("/health/youtube")
async def youtube_health_check():
    """Check if YouTube transcript extraction is working."""
    try:
        # Test with a known working video (Rick Roll - it always has captions!)
        test_video_id = "dQw4w9WgXcQ"
        
        enhanced_extractor = EnhancedYouTubeExtractor(webshare_manager)
        video_info = await enhanced_extractor.extract_video_info(test_video_id)
        
        # Try to fetch a small portion of transcript
        try:
            transcript = await fetch_transcript_with_enhanced_webshare(test_video_id)
            transcript_status = "working" if len(transcript) > 50 else "limited"
        except Exception as e:
            transcript_status = f"failed: {str(e)}"
        
        usage_stats = webshare_manager.get_usage_stats()
        
        return {
            "status": "healthy",
            "webshare_configured": bool(WEBSHARE_USERNAME and WEBSHARE_PASSWORD),
            "video_info_extraction": "working" if video_info else "failed",
            "transcript_extraction": transcript_status,
            "webshare_usage": usage_stats
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "webshare_configured": bool(WEBSHARE_USERNAME and WEBSHARE_PASSWORD)
        }

# Run the application with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # Check if required packages are installed
    try:
        import fastapi
        import uvicorn
        import PyPDF2
        import openai
        import genanki
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install the required packages using:")
        print("pip install fastapi uvicorn python-multipart PyPDF2 openai genanki")
        import sys
        sys.exit(1)

