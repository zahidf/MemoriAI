import os
from dotenv import load_dotenv
import tempfile
import traceback
import logging
from typing import List, Dict, Optional
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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
import asyncio
import requests
import json
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote, unquote

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info("Starting MemoriAI")

app = FastAPI(title="MemoriAI Anki Deck Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = f"Unexpected error: {str(exc)}"
    print(f"Global exception: {error_msg}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": error_msg},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_msg = f"Validation error: {str(exc)}"
    print(f"Validation exception: {error_msg}")
    return JSONResponse(
        status_code=422,
        content={"detail": error_msg},
    )

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
openai.api_key = DEEPSEEK_API_KEY
openai.base_url = "https://api.deepseek.com"

if not DEEPSEEK_API_KEY:
    print("Warning: DEEPSEEK_API_KEY environment variable is not set")

WEBSHARE_USERNAME = os.getenv('WEBSHARE_USERNAME')
WEBSHARE_PASSWORD = os.getenv('WEBSHARE_PASSWORD')

if not WEBSHARE_USERNAME or not WEBSHARE_PASSWORD:
    print("⚠️  WARNING: Webshare credentials not found!")
    print("📝 Set these environment variables:")
    print("   - WEBSHARE_USERNAME")
    print("   - WEBSHARE_PASSWORD")

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
    num_pairs: int = 0

class YoutubeRequest(BaseModel):
    url: str
    num_pairs: int = 0

class WorkingYouTubeExtractor:
    def __init__(self, webshare_username: str, webshare_password: str):
        self.webshare_username = webshare_username
        self.webshare_password = webshare_password
        
        self.proxy_endpoints = [
            "rotating-residential-proxies.all.webshare.io:8080",
            "premium-datacenter-proxies.all.webshare.io:8080",
            "static-residential-proxies.all.webshare.io:8080"
        ]
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def get_proxy_config(self) -> Dict[str, str]:
        endpoint = random.choice(self.proxy_endpoints)
        proxy_url = f"http://{self.webshare_username}:{self.webshare_password}@{endpoint}"
        
        return {
            'http': proxy_url,
            'https': proxy_url
        }
    
    async def extract_transcript(self, video_id: str, max_retries: int = 3) -> str:
        logger.info(f"🎯 Starting direct extraction for video: {video_id}")
        
        for attempt in range(max_retries):
            try:
                logger.info(f"🔄 Attempt {attempt + 1}/{max_retries}")
                
                proxies = self.get_proxy_config()
                
                if attempt > 0:
                    await asyncio.sleep(2 ** attempt)
                
                transcript = await self._extract_from_timedtext_api(video_id, proxies)
                if transcript:
                    logger.info(f"✅ Success with timedtext API: {len(transcript)} characters")
                    return transcript
                
                transcript = await self._extract_from_watch_page(video_id, proxies)
                if transcript:
                    logger.info(f"✅ Success with watch page: {len(transcript)} characters")
                    return transcript
                
                logger.warning(f"⚠️ Attempt {attempt + 1} failed, trying next method...")
                
            except Exception as e:
                logger.warning(f"⚠️ Attempt {attempt + 1} error: {str(e)}")
                if attempt == max_retries - 1:
                    raise e
        
        raise Exception("All extraction methods failed")
    
    async def _extract_from_timedtext_api(self, video_id: str, proxies: Dict[str, str]) -> Optional[str]:
        try:
            watch_url = f"https://www.youtube.com/watch?v={video_id}"
            
            response = requests.get(
                watch_url,
                headers=self.headers,
                proxies=proxies,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.warning(f"Watch page returned {response.status_code}")
                return None
            
            html_content = response.text
            
            patterns = [
                r'"captionTracks":\s*\[([^\]]+)\]',
                r'"captions":\s*\{[^}]*"playerCaptionsTracklistRenderer":\s*\{[^}]*"captionTracks":\s*\[([^\]]+)\]',
                r'\"baseUrl\":\"([^\"]*\/timedtext[^\"]*?)\"'
            ]
            
            caption_urls = []
            
            for pattern in patterns:
                matches = re.findall(pattern, html_content)
                for match in matches:
                    if 'timedtext' in match:
                        url = match.replace('\\u0026', '&').replace('\\/', '/')
                        if url.startswith('http'):
                            caption_urls.append(url)
            
            timedtext_pattern = r'https://www\.youtube\.com/api/timedtext[^\"]*'
            timedtext_urls = re.findall(timedtext_pattern, html_content.replace('\\/', '/'))
            caption_urls.extend(timedtext_urls)
            
            logger.info(f"Found {len(caption_urls)} potential caption URLs")
            
            for url in caption_urls[:5]:
                try:
                    if '&fmt=' not in url:
                        url += '&fmt=srv3'
                    
                    caption_response = requests.get(
                        url,
                        headers=self.headers,
                        proxies=proxies,
                        timeout=20
                    )
                    
                    if caption_response.status_code == 200:
                        transcript = self._parse_caption_response(caption_response.text)
                        if transcript and len(transcript) > 100:
                            return transcript
                            
                except Exception as e:
                    logger.warning(f"Caption URL failed: {str(e)}")
                    continue
            
            return None
            
        except Exception as e:
            logger.warning(f"Timedtext API extraction failed: {str(e)}")
            return None
    
    async def _extract_from_watch_page(self, video_id: str, proxies: Dict[str, str]) -> Optional[str]:
        try:
            watch_url = f"https://www.youtube.com/watch?v={video_id}"
            
            response = requests.get(
                watch_url,
                headers=self.headers,
                proxies=proxies,
                timeout=30
            )
            
            if response.status_code != 200:
                return None
            
            html_content = response.text
            
            player_response_pattern = r'var ytInitialPlayerResponse = ({.*?});'
            match = re.search(player_response_pattern, html_content)
            
            if match:
                try:
                    player_data = json.loads(match.group(1))
                    return self._extract_from_player_response(player_data, proxies)
                except json.JSONDecodeError:
                    pass
            
            transcript_patterns = [
                r'"transcriptText":"([^"]*)"',
                r'"caption":"([^"]*)"',
                r'"text":"([^"]*)"'
            ]
            
            for pattern in transcript_patterns:
                matches = re.findall(pattern, html_content)
                if matches and len(matches) > 10:
                    transcript = ' '.join(matches)
                    if len(transcript) > 200:
                        return transcript
            
            return None
            
        except Exception as e:
            logger.warning(f"Watch page extraction failed: {str(e)}")
            return None
    
    def _extract_from_player_response(self, player_data: Dict, proxies: Dict[str, str]) -> Optional[str]:
        try:
            captions = player_data.get('captions', {})
            caption_tracks = captions.get('playerCaptionsTracklistRenderer', {}).get('captionTracks', [])
            
            if not caption_tracks:
                return None
            
            selected_track = None
            for track in caption_tracks:
                lang_code = track.get('languageCode', '').lower()
                if lang_code.startswith('en') or lang_code == 'a.en':
                    selected_track = track
                    break
            
            if not selected_track and caption_tracks:
                selected_track = caption_tracks[0]
            
            if selected_track:
                base_url = selected_track.get('baseUrl')
                if base_url:
                    if '&fmt=' not in base_url:
                        base_url += '&fmt=srv3'
                    
                    try:
                        response = requests.get(
                            base_url,
                            headers=self.headers,
                            proxies=proxies,
                            timeout=20
                        )
                        
                        if response.status_code == 200:
                            return self._parse_caption_response(response.text)
                    except Exception as e:
                        logger.warning(f"Caption track request failed: {str(e)}")
            
            return None
            
        except Exception as e:
            logger.warning(f"Player response extraction failed: {str(e)}")
            return None
    
    def _parse_caption_response(self, content: str) -> Optional[str]:
        try:
            if content.strip().startswith('<'):
                try:
                    root = ET.fromstring(content)
                    transcript_parts = []
                    
                    for elem in root.iter():
                        if elem.tag in ['text', 'p'] and elem.text:
                            text = elem.text.strip()
                            text = re.sub(r'\[.*?\]', '', text)
                            text = re.sub(r'\s+', ' ', text)
                            if text and len(text) > 2:
                                transcript_parts.append(text)
                    
                    if transcript_parts:
                        full_transcript = ' '.join(transcript_parts)
                        full_transcript = re.sub(r'\[Music\]|\[Applause\]|\[Laughter\]', '', full_transcript)
                        full_transcript = re.sub(r'\s+', ' ', full_transcript).strip()
                        
                        return full_transcript if len(full_transcript) > 50 else None
                        
                except ET.ParseError:
                    pass
            
            try:
                data = json.loads(content)
                if isinstance(data, dict) and 'events' in data:
                    transcript_parts = []
                    for event in data['events']:
                        if 'segs' in event:
                            for seg in event['segs']:
                                if 'utf8' in seg:
                                    transcript_parts.append(seg['utf8'])
                    
                    if transcript_parts:
                        return ' '.join(transcript_parts)
            except json.JSONDecodeError:
                pass
            
            if len(content) > 100:
                clean_text = re.sub(r'<[^>]+>', ' ', content)
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                
                if len(clean_text) > 100:
                    return clean_text
            
            return None
            
        except Exception as e:
            logger.warning(f"Caption parsing failed: {str(e)}")
            return None

working_extractor = None

def initialize_working_extractor():
    global working_extractor
    
    if not WEBSHARE_USERNAME or not WEBSHARE_PASSWORD:
        raise Exception("WebShare credentials not configured")
    
    working_extractor = WorkingYouTubeExtractor(WEBSHARE_USERNAME, WEBSHARE_PASSWORD)
    logger.info(f"✅ Working YouTube extractor initialized for user: {WEBSHARE_USERNAME}")

async def fetch_transcript_that_actually_works(video_id: str) -> str:
    global working_extractor
    
    if not working_extractor:
        initialize_working_extractor()
    
    logger.info(f"🎬 Starting working extraction for video: {video_id}")
    
    try:
        transcript = await working_extractor.extract_transcript(video_id)
        
        if transcript and len(transcript.strip()) > 100:
            logger.info(f"✅ Successfully extracted {len(transcript)} characters")
            return transcript
        else:
            raise Exception("Extracted transcript was too short or empty")
            
    except Exception as e:
        logger.error(f"❌ Working extractor failed: {str(e)}")
        raise Exception(f"Could not extract transcript: {str(e)}")

tasks_status = {}

@app.on_event("startup")
async def setup_periodic_cleanup():
    import asyncio
    
    async def cleanup_old_tasks():
        while True:
            try:
                current_time = time.time()
                tasks_to_remove = []
                
                for task_id, task_data in tasks_status.items():
                    if not task_data.get("timestamp"):
                        task_data["timestamp"] = current_time
                        continue
                    
                    if current_time - task_data["timestamp"] > 24 * 60 * 60:
                        tasks_to_remove.append(task_id)
                        
                        if task_data.get("temp_dir") and os.path.exists(task_data["temp_dir"]):
                            try:
                                import shutil
                                shutil.rmtree(task_data["temp_dir"], ignore_errors=True)
                            except Exception as e:
                                print(f"Error cleaning up files for task {task_id}: {str(e)}")
                
                for task_id in tasks_to_remove:
                    del tasks_status[task_id]
                
                await asyncio.sleep(60 * 60)
            except Exception as e:
                print(f"Error in periodic cleanup: {str(e)}")
                await asyncio.sleep(10 * 60)
    
    asyncio.create_task(cleanup_old_tasks())

def add_timestamp_to_task(task_id: str):
    if task_id in tasks_status:
        tasks_status[task_id]["timestamp"] = time.time()

@app.get("/", response_class=HTMLResponse)
async def get_html():
    return Path("ankiFrontEnd.html").read_text()

@app.post("/upload/pdf", response_model=ProcessingStatus)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    num_pairs: Optional[int] = None
):
    try:
        task_id = f"task_{random.randint(10000, 99999)}"
        
        logger.info(f"Reading file content for {file.filename}")
        file_content = await file.read()
        file_size = len(file_content)
        
        logger.info(f"Read {file_size} bytes from {file.filename}")
        
        if file_size > 20 * 1024 * 1024:
            raise ValueError("PDF file is too large (over 20MB). Please upload a smaller file.")
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(file_content)
        temp_file.close()
        
        try:
            import fitz
            doc = fitz.open(temp_file.name)
            page_count = len(doc)
            doc.close()
        except Exception as e:
            logger.warning(f"Could not count PDF pages: {str(e)}")
            page_count = max(1, file_size // (100 * 1024))
        finally:
            os.unlink(temp_file.name)
        
        estimated_count = estimate_card_count("pdf", page_count)
        
        if num_pairs is None or num_pairs <= 0:
            num_pairs = estimated_count
        else:
            num_pairs = min(50, max(1, int(num_pairs)))
        
        logger.info(f"Estimated {estimated_count} cards for {page_count} pages, using {num_pairs}")
        
        tasks_status[task_id] = {
            "status": "processing",
            "progress": 0.0,
            "message": f"Initializing (generating {num_pairs} Q&A pairs)",
            "file_path": None,
            "qa_pairs": None,
            "estimated_count": estimated_count
        }
        
        add_timestamp_to_task(task_id)
        
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
    temp_dir = None
    try:
        logger.info(f"Starting PDF processing for task {task_id}")
        
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, "input.pdf")
        
        tasks_status[task_id].update({
            "temp_dir": temp_dir,
            "file_path": file_path,
            "progress": 0.1,
            "message": "Writing PDF file"
        })
        
        with open(file_path, "wb") as temp_file:
            temp_file.write(file_content)
        logger.info(f"Successfully wrote {len(file_content)} bytes to {file_path}")
        
        tasks_status[task_id].update({
            "progress": 0.3,
            "message": "Extracting text from PDF"
        })
        
        try:
            logger.info("Extracting text using pymupdf4llm")
            text = pymupdf4llm.to_markdown(file_path)
            logger.info(f"Successfully extracted {len(text)} characters of text")
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}", exc_info=True)
            tasks_status[task_id].update({
                "status": "failed",
                "message": f"Could not extract text from the PDF: {str(e)}"
            })
            return
        
        if not text or len(text.strip()) < 50:
            logger.warning(f"Extracted text too short: {len(text.strip() if text else 0)} characters")
            tasks_status[task_id].update({
                "status": "failed",
                "message": "Could not extract enough text from the PDF. The file might be scanned images or protected."
            })
            return
        
        tasks_status[task_id].update({
            "progress": 0.5,
            "message": "Generating QA pairs"
        })
        
        try:
            logger.info(f"Generating QA pairs from {len(text)} characters of text")
            qa_pairs = generate_qa_pairs(text, num_pairs)
            logger.info(f"Successfully generated {len(qa_pairs)} QA pairs")
            
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
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {str(e)}")

def generate_qa_pairs(text: str, num_pairs: int) -> List[Dict[str, str]]:
    try:
        if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your-deepseek-api-key-here":
            logger.error("DeepSeek API key is not set")
            return [{"question": "Error: DeepSeek API key is not set", 
                    "answer": "Please set your DeepSeek API key in the main.py file."}]
        
        if not text or len(text.strip()) < 10:
            logger.warning("Input text is too short or empty")
            return [{"question": "Error: Input text is too short", 
                    "answer": "Please provide more text to generate meaningful questions."}]
        
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
        
        client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        
        max_chars = 12000
        if len(text) > max_chars:
            logger.warning(f"Text too long ({len(text)} chars), truncating to {max_chars} chars")
            text = text[:max_chars] + "..."
        
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
        
        import json
        import re
        
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = content
        
        json_str = re.sub(r'```.*', '', json_str)
        json_str = re.sub(r'^.*?\[', '[', json_str.strip(), count=1, flags=re.DOTALL)
        if not json_str.startswith('['):
            json_str = '[' + json_str
        if not json_str.endswith(']'):
            json_str = json_str + ']'
        
        try:
            qa_pairs = json.loads(json_str)
            logger.info(f"Successfully parsed {len(qa_pairs)} QA pairs from response")
            
            valid_pairs = []
            for pair in qa_pairs:
                if isinstance(pair, dict) and 'question' in pair and 'answer' in pair:
                    valid_pairs.append({
                        'question': pair['question'],
                        'answer': pair['answer']
                    })
            
            logger.info(f"Validated {len(valid_pairs)} QA pairs")
            
            if len(valid_pairs) > num_pairs:
                logger.warning(f"Got {len(valid_pairs)} pairs but only requested {num_pairs}, trimming list")
                valid_pairs = valid_pairs[:num_pairs]
            
            if len(valid_pairs) < num_pairs:
                logger.warning(f"Requested {num_pairs} pairs but only got {len(valid_pairs)}")
            
            return valid_pairs
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return [{"question": "Error parsing API response", 
                    "answer": "The API returned a response that couldn't be parsed as JSON."}]
            
    except Exception as e:
        logger.error(f"Error in generate_qa_pairs: {str(e)}", exc_info=True)
        return [{"question": "Error generating QA pairs", 
                "answer": f"An error occurred: {str(e)}"}]

def create_anki_deck(title: str, qa_pairs: List[Dict[str, str]]) -> str:
    model_id = random.randrange(1 << 30, 1 << 31)
    deck_id = random.randrange(1 << 30, 1 << 31)

    logger.info(f"Creating Anki deck with title: {title}")
    
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
    
    deck = genanki.Deck(deck_id, title)
    
    for qa in qa_pairs:
        note = genanki.Note(
            model=model,
            fields=[qa['question'], qa['answer']]
        )
        deck.add_note(note)
    
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, f"{title.replace(' ', '_')}.apkg")
    
    genanki.Package(deck).write_to_file(output_path)
    
    return output_path

@app.get("/status/{task_id}", response_model=ProcessingStatus)
async def get_status(task_id: str):
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
    task_id = f"deck_{random.randint(10000, 99999)}"
    
    tasks_status[task_id] = {
        "status": "processing",
        "progress": 0.0,
        "message": "Starting deck generation",
        "file_path": None,
        "timestamp": time.time()
    }
    
    background_tasks.add_task(create_deck_task, task_id, request.title, request.qa_pairs)
    
    return ProcessingStatus(task_id=task_id, status="processing", progress=0.0)

async def create_deck_task(task_id: str, title: str, qa_pairs: List[QAPair]):
    temp_dir = None
    try:
        qa_dict_pairs = [{"question": qa.question, "answer": qa.answer} for qa in qa_pairs]
        
        tasks_status[task_id]["progress"] = 0.3
        tasks_status[task_id]["message"] = "Creating Anki deck"
        
        file_path = create_anki_deck(title, qa_dict_pairs)
        temp_dir = os.path.dirname(file_path)
        
        tasks_status[task_id]["progress"] = 1.0
        tasks_status[task_id]["status"] = "completed"
        tasks_status[task_id]["message"] = "Deck generation complete"
        tasks_status[task_id]["file_path"] = file_path
        
    except Exception as e:
        tasks_status[task_id]["status"] = "failed"
        tasks_status[task_id]["message"] = f"Error: {str(e)}"
        
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as cleanup_error:
                print(f"Warning: Failed to clean up temporary files after error: {cleanup_error}")

@app.get("/download/{task_id}")
async def download_deck(task_id: str, background_tasks: BackgroundTasks):
    if task_id not in tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    status = tasks_status[task_id]
    if status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Deck generation not completed")
    
    if not status.get("file_path") or not os.path.exists(status["file_path"]):
        raise HTTPException(status_code=404, detail="Deck file not found")
    
    file_path = status["file_path"]
    filename = os.path.basename(file_path)
    
    def cleanup_temp_files():
        try:
            import time
            time.sleep(60)
            
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
    task_id = f"task_{random.randint(10000, 99999)}"
    
    text_length = len(request.text)
    estimated_count = estimate_card_count("text", text_length)
    
    num_pairs = request.num_pairs if request.num_pairs > 0 else estimated_count
    num_pairs = min(50, max(1, num_pairs))
    
    logger.info(f"Estimated {estimated_count} cards for {text_length} chars, using {num_pairs}")
    
    tasks_status[task_id] = {
        "status": "processing",
        "progress": 0.0,
        "message": "Starting text processing",
        "qa_pairs": None,
        "timestamp": time.time(),
        "estimated_count": estimated_count
    }
    
    background_tasks.add_task(process_text_task, task_id, request.text, num_pairs)
    
    return ProcessingStatus(
        task_id=task_id, 
        status="processing", 
        progress=0.0,
        estimated_count=estimated_count
    )

async def process_text_task(task_id: str, text: str, num_pairs: int):
    try:
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
            
        tasks_status[task_id].update({
            "progress": 0.3,
            "message": f"Processing text (generating {num_pairs} Q&A pairs)"
        })
        
        if len(text.strip()) < 100:
            tasks_status[task_id].update({
                "status": "failed",
                "message": "The provided text is too short. Please provide more content to generate meaningful questions."
            })
            return
        
        logger.info(f"Generating {num_pairs} Q&A pairs from {len(text)} characters of text")
        qa_pairs = generate_qa_pairs(text, num_pairs)
        logger.info(f"Successfully generated {len(qa_pairs)} Q&A pairs")
        
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
    try:
        if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your-deepseek-api-key-here":
            return {"status": "error", "message": "DeepSeek API key is not set. Please update the DEEPSEEK_API_KEY in main.py."}
        
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
    if content_type == "pdf":
        estimated_count = max(5, round(content_size * 2.5))
    elif content_type == "text":
        word_count = content_size / 6
        estimated_count = max(5, round(word_count / 150))
    elif content_type == "youtube":
        minutes = content_size / 60
        estimated_count = max(5, round(minutes))
    else:
        estimated_count = 10
    
    estimated_count = round(estimated_count / 5) * 5
    
    return min(estimated_count, 50)

def extract_youtube_id(url: str) -> str:
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

@app.post("/process/youtube", response_model=ProcessingStatus)
async def process_youtube_with_working_method(request: YoutubeRequest, background_tasks: BackgroundTasks):
    task_id = f"task_{random.randint(10000, 99999)}"
    
    logger.info(f"🎬 New working YouTube request: {request.url}")
    
    estimated_duration_minutes = 10
    estimated_count = estimate_card_count("youtube", estimated_duration_minutes * 60)
    
    num_pairs = request.num_pairs if request.num_pairs > 0 else estimated_count
    num_pairs = min(50, max(1, num_pairs))
    
    tasks_status[task_id] = {
        "status": "processing",
        "progress": 0.0,
        "message": "Starting working YouTube transcript extraction...",
        "qa_pairs": None,
        "timestamp": time.time(),
        "estimated_count": estimated_count
    }
    
    background_tasks.add_task(process_youtube_task_with_working_extraction, task_id, request.url, num_pairs)
    
    return ProcessingStatus(
        task_id=task_id, 
        status="processing", 
        progress=0.0,
        message="Processing with working extraction method that bypasses IP blocks",
        estimated_count=estimated_count
    )

async def process_youtube_task_with_working_extraction(task_id: str, url: str, num_pairs: int):
    try:
        logger.info(f"🎬 Starting working YouTube processing for task {task_id}")
        
        num_pairs = max(1, min(50, int(num_pairs)))
        
        tasks_status[task_id].update({
            "progress": 0.1,
            "message": "Extracting video ID..."
        })
        
        try:
            video_id = extract_youtube_id(url)
            logger.info(f"✅ Video ID: {video_id}")
        except ValueError as e:
            tasks_status[task_id].update({
                "status": "failed", 
                "message": f"Invalid YouTube URL: {str(e)}"
            })
            return
        
        tasks_status[task_id].update({
            "progress": 0.3,
            "message": "Extracting transcript with working method..."
        })
        
        try:
            transcript_text = await fetch_transcript_that_actually_works(video_id)
            logger.info(f"✅ Working extraction success: {len(transcript_text)} characters")
            
        except Exception as e:
            logger.error(f"❌ Working extraction failed: {str(e)}")
            tasks_status[task_id].update({
                "status": "failed",
                "message": f"Could not extract transcript: {str(e)}. Please ensure the video has captions enabled."
            })
            return
        
        tasks_status[task_id].update({
            "progress": 0.7,
            "message": f"Generating {num_pairs} Q&A pairs..."
        })
        
        logger.info(f"🤖 Generating {num_pairs} Q&A pairs from {len(transcript_text)} characters")
        qa_pairs = generate_qa_pairs(transcript_text, num_pairs)
        logger.info(f"✅ Generated {len(qa_pairs)} Q&A pairs")
        
        tasks_status[task_id].update({
            "progress": 1.0,
            "status": "completed",
            "message": f"Processing complete ({len(qa_pairs)} Q&A pairs generated)",
            "qa_pairs": qa_pairs
        })
        
        logger.info(f"🎉 Working YouTube processing completed for task {task_id}")
        
    except Exception as e:
        logger.error(f"💥 Unexpected error: {str(e)}", exc_info=True)
        tasks_status[task_id].update({
            "status": "failed",
            "message": f"Unexpected error: {str(e)}"
        })

@app.get("/debug/working-extraction/{video_id}")
async def test_working_extraction(video_id: str):
    try:
        logger.info(f"🧪 Testing working extraction for: {video_id}")
        
        transcript = await fetch_transcript_that_actually_works(video_id)
        
        return {
            "status": "success",
            "video_id": video_id,
            "transcript_length": len(transcript),
            "sample": transcript[:200] + "..." if len(transcript) > 200 else transcript,
            "method": "working_extractor"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "video_id": video_id,
            "error": str(e),
            "method": "working_extractor"
        }

@app.get("/youtube/info/{video_id}")
async def get_youtube_video_info(video_id: str):
    try:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        
        response = requests.get(oembed_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "status": "success",
                "video_id": video_id,
                "info": {
                    'title': data.get('title', f'YouTube Video ({video_id})'),
                    'author': data.get('author_name', 'Unknown'),
                    'duration': data.get('duration', 'Unknown')
                }
            }
        else:
            return {
                "status": "error",
                "video_id": video_id,
                "error": f"Failed to get video info: HTTP {response.status_code}"
            }
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        return {
            "status": "error",
            "video_id": video_id,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)