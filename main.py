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
DECODO_USERNAME = os.getenv('DECODO_USERNAME')
DECODO_PASSWORD = os.getenv('DECODO_PASSWORD')

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

class UltimateYouTubeExtractor:
    def __init__(self, decodo_username: str = None, decodo_password: str = None):
        self.decodo_username = decodo_username
        self.decodo_password = decodo_password
        
        # Multiple proxy endpoints
        self.proxy_endpoint = "gate.decodo.com:7000"
        
        # Realistic browser headers with rotation
        self.header_sets = [
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            },
            {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Sec-Ch-Ua': '"Google Chrome";v="120", "Chromium";v="120", "Not_A Brand";v="24"',
            },
            {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Sec-Ch-Ua': '"Chromium";v="120", "Not_A Brand";v="8", "Google Chrome";v="120"',
            }
        ]
        
        # Add common headers for all sets
        for header_set in self.header_sets:
            header_set.update({
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Ch-Ua-Mobile': '?0',
                'Sec-Ch-Ua-Platform': '"Windows"',
            })
    
    def get_headers(self) -> Dict[str, str]:
        """Get random headers to appear more human-like"""
        return random.choice(self.header_sets)
    
    def get_proxy_config(self, country_code: str = "US", session_id: str = None) -> Optional[Dict[str, str]]:
        """Get Decodo proxy configuration"""
        if not self.decodo_username or not self.decodo_password:
            return None
        
        username_parts = [f"user-{self.decodo_username}"]
        
        if country_code:
            username_parts.append(f"country-{country_code.lower()}")
        
        if session_id:
            username_parts.append(f"session-{session_id}")
            username_parts.append("sessionduration-30")
        
        proxy_username = "-".join(username_parts)
        proxy_url = f"http://{proxy_username}:{self.decodo_password}@{self.proxy_endpoint}"
        
        return {
            'http': proxy_url,
            'https': proxy_url
        }
    
    async def extract_transcript(self, video_id: str) -> str:
        """Ultimate transcript extraction with multiple fallback methods"""
        logger.info(f"🎯 Starting ultimate extraction for video: {video_id}")
        
        # Method 1: Try youtube-transcript-api with proxy (if available)
        if self.decodo_username and self.decodo_password:
            try:
                transcript = await self._extract_with_youtube_transcript_api_and_proxy(video_id)
                if transcript:
                    return transcript
            except Exception as e:
                logger.warning(f"Method 1 (API + proxy) failed: {e}")
        
        # Method 2: Try youtube-transcript-api without proxy
        try:
            transcript = await self._extract_with_youtube_transcript_api_direct(video_id)
            if transcript:
                return transcript
        except Exception as e:
            logger.warning(f"Method 2 (API direct) failed: {e}")
        
        # Method 3: Browser simulation with proxy (if available)
        if self.decodo_username and self.decodo_password:
            try:
                transcript = await self._extract_via_browser_simulation_with_proxy(video_id)
                if transcript:
                    return transcript
            except Exception as e:
                logger.warning(f"Method 3 (browser sim + proxy) failed: {e}")
        
        # Method 4: Browser simulation without proxy
        try:
            transcript = await self._extract_via_browser_simulation_direct(video_id)
            if transcript:
                return transcript
        except Exception as e:
            logger.warning(f"Method 4 (browser sim direct) failed: {e}")
        
        # Method 5: Legacy timedtext API attempts
        try:
            transcript = await self._extract_via_legacy_methods(video_id)
            if transcript:
                return transcript
        except Exception as e:
            logger.warning(f"Method 5 (legacy) failed: {e}")
        
        # Method 6: YouTube embed page extraction
        try:
            transcript = await self._extract_via_embed_page(video_id)
            if transcript:
                return transcript
        except Exception as e:
            logger.warning(f"Method 6 (embed page) failed: {e}")
        
        # Method 7: Mobile YouTube page
        try:
            transcript = await self._extract_via_mobile_page(video_id)
            if transcript:
                return transcript
        except Exception as e:
            logger.warning(f"Method 7 (mobile page) failed: {e}")
        
        # If all methods fail, provide detailed troubleshooting
        raise Exception(self._get_troubleshooting_message(video_id))
    
    async def _extract_with_youtube_transcript_api_and_proxy(self, video_id: str) -> Optional[str]:
        """Method 1: youtube-transcript-api with Decodo proxy"""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api.proxies import GenericProxyConfig
            
            session_id = f"yt_api_{video_id}_{int(time.time())}"
            proxies = self.get_proxy_config("US", session_id)
            
            if not proxies:
                return None
            
            proxy_config = GenericProxyConfig(
                http_url=proxies['http'],
                https_url=proxies['https']
            )
            
            api = YouTubeTranscriptApi(proxy_config=proxy_config)
            transcript_list = api.list_transcripts(video_id)
            
            # Get best available transcript
            transcript = self._get_best_transcript(transcript_list)
            if not transcript:
                return None
            
            transcript_data = transcript.fetch()
            return self._process_transcript_data(transcript_data)
            
        except ImportError:
            logger.warning("youtube-transcript-api not installed")
            return None
        except Exception as e:
            logger.warning(f"YouTube API with proxy failed: {e}")
            return None
    
    async def _extract_with_youtube_transcript_api_direct(self, video_id: str) -> Optional[str]:
        """Method 2: youtube-transcript-api without proxy"""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            
            api = YouTubeTranscriptApi()
            transcript_list = api.list_transcripts(video_id)
            
            transcript = self._get_best_transcript(transcript_list)
            if not transcript:
                return None
            
            transcript_data = transcript.fetch()
            return self._process_transcript_data(transcript_data)
            
        except ImportError:
            return None
        except Exception as e:
            logger.warning(f"YouTube API direct failed: {e}")
            return None
    
    async def _extract_via_browser_simulation_with_proxy(self, video_id: str) -> Optional[str]:
        """Method 3: Simulate real browser with proxy"""
        countries = ["US", "GB", "CA", "AU"]
        
        for country in countries:
            try:
                session_id = f"browser_{video_id}_{country}_{int(time.time())}"
                proxies = self.get_proxy_config(country, session_id)
                headers = self.get_headers()
                
                # First get the main page
                watch_url = f"https://www.youtube.com/watch?v={video_id}"
                response = requests.get(
                    watch_url,
                    headers=headers,
                    proxies=proxies,
                    timeout=30,
                    allow_redirects=True
                )
                
                if response.status_code == 200:
                    transcript = await self._extract_from_watch_page_content(response.text, video_id, proxies, headers)
                    if transcript:
                        logger.info(f"✅ Browser simulation with proxy success from {country}")
                        return transcript
                
                # Add delay between countries
                await asyncio.sleep(random.uniform(1, 3))
                
            except Exception as e:
                logger.debug(f"Browser sim with proxy failed for {country}: {e}")
                continue
        
        return None
    
    async def _extract_via_browser_simulation_direct(self, video_id: str) -> Optional[str]:
        """Method 4: Simulate real browser without proxy"""
        try:
            headers = self.get_headers()
            watch_url = f"https://www.youtube.com/watch?v={video_id}"
            
            response = requests.get(
                watch_url,
                headers=headers,
                timeout=30,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                return await self._extract_from_watch_page_content(response.text, video_id, None, headers)
            
        except Exception as e:
            logger.warning(f"Browser simulation direct failed: {e}")
        
        return None
    
    async def _extract_via_legacy_methods(self, video_id: str) -> Optional[str]:
        """Method 5: Try legacy timedtext API endpoints"""
        
        # Different timedtext endpoints to try
        endpoints = [
            "https://www.youtube.com/api/timedtext",
            "https://video.google.com/timedtext",
        ]
        
        # Different parameter combinations
        param_sets = [
            {"v": video_id, "lang": "en", "fmt": "json3"},
            {"v": video_id, "lang": "en-US", "fmt": "json3"},
            {"v": video_id, "lang": "a.en", "fmt": "json3"},
            {"v": video_id, "lang": "en", "fmt": "srv3"},
            {"v": video_id, "lang": "en", "fmt": "vtt"},
            {"v": video_id, "lang": "en"},  # No format specified
        ]
        
        headers = self.get_headers()
        proxies = self.get_proxy_config() if self.decodo_username else None
        
        for endpoint in endpoints:
            for params in param_sets:
                try:
                    response = requests.get(
                        endpoint,
                        params=params,
                        headers=headers,
                        proxies=proxies,
                        timeout=15
                    )
                    
                    if response.status_code == 200 and response.content:
                        transcript = self._parse_transcript_content(response.text)
                        if transcript and len(transcript) > 100:
                            logger.info(f"✅ Legacy method success with {endpoint}")
                            return transcript
                    
                    # Small delay between requests
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.debug(f"Legacy method failed for {endpoint} with {params}: {e}")
                    continue
        
        return None
    
    async def _extract_via_embed_page(self, video_id: str) -> Optional[str]:
        """Method 6: Try YouTube embed page"""
        try:
            headers = self.get_headers()
            proxies = self.get_proxy_config() if self.decodo_username else None
            
            embed_url = f"https://www.youtube.com/embed/{video_id}"
            
            response = requests.get(
                embed_url,
                headers=headers,
                proxies=proxies,
                timeout=20
            )
            
            if response.status_code == 200:
                return await self._extract_from_watch_page_content(response.text, video_id, proxies, headers)
            
        except Exception as e:
            logger.warning(f"Embed page extraction failed: {e}")
        
        return None
    
    async def _extract_via_mobile_page(self, video_id: str) -> Optional[str]:
        """Method 7: Try mobile YouTube page"""
        try:
            mobile_headers = self.get_headers()
            mobile_headers['User-Agent'] = 'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1'
            
            proxies = self.get_proxy_config() if self.decodo_username else None
            
            mobile_url = f"https://m.youtube.com/watch?v={video_id}"
            
            response = requests.get(
                mobile_url,
                headers=mobile_headers,
                proxies=proxies,
                timeout=20
            )
            
            if response.status_code == 200:
                return await self._extract_from_watch_page_content(response.text, video_id, proxies, mobile_headers)
            
        except Exception as e:
            logger.warning(f"Mobile page extraction failed: {e}")
        
        return None
    
    async def _extract_from_watch_page_content(self, html_content: str, video_id: str, proxies: Optional[Dict[str, str]], headers: Dict[str, str]) -> Optional[str]:
        """Extract transcript from YouTube page HTML content"""
        try:
            # Look for various patterns in the HTML
            patterns = [
                r'"captionTracks":\s*(\[[^\]]+\])',
                r'"captions":\s*\{[^}]*"playerCaptionsTracklistRenderer":\s*\{[^}]*"captionTracks":\s*(\[[^\]]+\])',
                r'\"baseUrl\":\"(https://[^\"]*?timedtext[^\"]*?)\"',
                r'"timedtext"[^}]*?"baseUrl":\s*"([^"]+)"',
            ]
            
            caption_urls = []
            
            for pattern in patterns:
                matches = re.findall(pattern, html_content)
                for match in matches:
                    if match.startswith('['):
                        try:
                            tracks = json.loads(match)
                            for track in tracks:
                                if isinstance(track, dict) and 'baseUrl' in track:
                                    url = track['baseUrl'].replace('\\u0026', '&').replace('\\/', '/')
                                    caption_urls.append(url)
                        except json.JSONDecodeError:
                            continue
                    elif 'timedtext' in match:
                        url = match.replace('\\u0026', '&').replace('\\/', '/')
                        caption_urls.append(url)
            
            # Try each caption URL
            for url in caption_urls[:5]:  # Limit to first 5
                try:
                    if not url.startswith('http'):
                        continue
                    
                    # Ensure we get a parseable format
                    if '&fmt=' not in url:
                        url += '&fmt=json3'
                    
                    caption_response = requests.get(
                        url,
                        headers=headers,
                        proxies=proxies,
                        timeout=15
                    )
                    
                    if caption_response.status_code == 200 and caption_response.content:
                        transcript = self._parse_transcript_content(caption_response.text)
                        if transcript and len(transcript) > 100:
                            return transcript
                    
                    await asyncio.sleep(0.5)  # Small delay between requests
                    
                except Exception as e:
                    logger.debug(f"Caption URL failed: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.warning(f"Watch page content extraction failed: {e}")
            return None
    
    def _get_best_transcript(self, transcript_list) -> Optional[object]:
        """Get the best available transcript from the list"""
        # Priority order: manual English > auto English > any manual > any auto
        try:
            return transcript_list.find_manually_created_transcript(['en', 'en-US', 'en-GB'])
        except:
            try:
                return transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
            except:
                try:
                    # Get any manual transcript and translate
                    manual_transcripts = [t for t in transcript_list if not t.is_generated]
                    if manual_transcripts:
                        return manual_transcripts[0].translate('en')
                except:
                    try:
                        # Last resort: any auto transcript and translate
                        auto_transcripts = [t for t in transcript_list if t.is_generated]
                        if auto_transcripts:
                            return auto_transcripts[0].translate('en')
                    except:
                        pass
        return None
    
    def _process_transcript_data(self, transcript_data: List[Dict]) -> Optional[str]:
        """Process transcript data from youtube-transcript-api"""
        try:
            text_parts = []
            
            for entry in transcript_data:
                text = entry.get('text', '').strip()
                if text and not any(noise in text.lower() for noise in [
                    '[music]', '[applause]', '[laughter]', '♪', '♫', 
                    '[sound effects]', '[silence]', '[inaudible]'
                ]):
                    text_parts.append(text)
            
            if not text_parts:
                return None
            
            full_transcript = ' '.join(text_parts)
            return self._clean_transcript(full_transcript)
            
        except Exception as e:
            logger.warning(f"Transcript data processing failed: {e}")
            return None
    
    def _parse_transcript_content(self, content: str) -> Optional[str]:
        """Parse transcript from various formats"""
        try:
            if not content or len(content.strip()) < 10:
                return None
            
            # JSON3 format
            if content.strip().startswith('{'):
                try:
                    data = json.loads(content)
                    if 'events' in data:
                        text_parts = []
                        for event in data['events']:
                            if 'segs' in event:
                                for seg in event['segs']:
                                    if 'utf8' in seg:
                                        text = seg['utf8'].strip()
                                        if text and text not in ['[Music]', '[Applause]', '[Laughter]']:
                                            text_parts.append(text)
                        
                        if text_parts:
                            full_transcript = ' '.join(text_parts)
                            return self._clean_transcript(full_transcript)
                            
                except json.JSONDecodeError:
                    pass
            
            # XML format
            if content.strip().startswith('<'):
                try:
                    root = ET.fromstring(content)
                    text_parts = []
                    
                    for elem in root.iter():
                        if elem.tag in ['text', 'p'] and elem.text:
                            text = elem.text.strip()
                            if text and len(text) > 2:
                                text_parts.append(text)
                    
                    if text_parts:
                        full_transcript = ' '.join(text_parts)
                        return self._clean_transcript(full_transcript)
                        
                except ET.ParseError:
                    pass
            
            # VTT format
            if 'WEBVTT' in content:
                try:
                    lines = content.split('\n')
                    text_parts = []
                    for line in lines:
                        line = line.strip()
                        if ('-->' not in line and line and 
                            not line.startswith('WEBVTT') and 
                            not line.isdigit() and
                            not line.startswith('NOTE')):
                            line = re.sub(r'<[^>]+>', '', line)
                            line = re.sub(r'\[.*?\]', '', line)
                            line = line.strip()
                            if line and len(line) > 2:
                                text_parts.append(line)
                    
                    if text_parts:
                        full_transcript = ' '.join(text_parts)
                        return self._clean_transcript(full_transcript)
                        
                except Exception:
                    pass
            
            return None
            
        except Exception as e:
            logger.warning(f"Transcript content parsing failed: {e}")
            return None
    
    def _clean_transcript(self, transcript: str) -> Optional[str]:
        """Clean and normalize transcript text"""
        try:
            if not transcript:
                return None
            
            # Remove bracketed content
            transcript = re.sub(r'\[.*?\]', '', transcript)
            # Remove music symbols
            transcript = re.sub(r'♪.*?♪', '', transcript)
            # Remove multiple whitespace
            transcript = re.sub(r'\s+', ' ', transcript).strip()
            
            # Check if transcript is long enough
            if len(transcript) < 50:
                return None
            
            return transcript
            
        except Exception as e:
            logger.warning(f"Transcript cleaning failed: {e}")
            return None
    
    def _get_troubleshooting_message(self, video_id: str) -> str:
        """Generate detailed troubleshooting message"""
        return f"""
All transcript extraction methods failed for video {video_id}.

Possible causes:
1. Video doesn't have captions/transcripts available
2. Video is private, restricted, or deleted
3. YouTube has enhanced blocking for this specific video
4. Geographic restrictions preventing access
5. Temporary YouTube API issues

Troubleshooting steps:
1. Verify the video has captions by checking manually: https://www.youtube.com/watch?v={video_id}
2. Try a different video ID to test if the issue is video-specific
3. Check if the video is accessible from your location
4. Ensure your proxy credentials are correct (if using Decodo)
5. Try again in a few minutes in case of temporary blocks

Methods attempted:
- YouTube Transcript API with proxy
- YouTube Transcript API direct
- Browser simulation with proxy
- Browser simulation direct  
- Legacy timedtext API endpoints
- YouTube embed page extraction
- Mobile YouTube page extraction

If the video definitely has captions and this still fails, YouTube may have 
implemented new anti-bot measures that require more sophisticated bypassing.
"""


# Updated integration for your main.py
ultimate_extractor = None

def initialize_working_extractor():
    """Initialize the ultimate extractor"""
    initialize_ultimate_extractor()

def initialize_ultimate_extractor():
    global ultimate_extractor
    
    ultimate_extractor = UltimateYouTubeExtractor(
        decodo_username=DECODO_USERNAME,
        decodo_password=DECODO_PASSWORD
    )
    logger.info("✅ Ultimate YouTube extractor initialized")


async def fetch_transcript_that_actually_works(video_id: str) -> str:
    global ultimate_extractor
    
    if not ultimate_extractor:
        initialize_ultimate_extractor()
    
    logger.info(f"🎬 Starting ultimate extraction for video: {video_id}")
    
    try:
        transcript = await ultimate_extractor.extract_transcript(video_id)
        
        if transcript and len(transcript.strip()) > 100:
            logger.info(f"✅ Successfully extracted {len(transcript)} characters")
            return transcript
        else:
            raise Exception("Extracted transcript was too short or empty")
            
    except Exception as e:
        logger.error(f"❌ Ultimate extractor failed: {str(e)}")
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
    

class DecodoYouTubeExtractor:
    def __init__(self, decodo_username: str, decodo_password: str):
        self.decodo_username = decodo_username
        self.decodo_password = decodo_password
        
        # Decodo's single endpoint for all residential proxies
        self.proxy_endpoint = "gate.decodo.com:7000"
        
        # Realistic browser headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
        }
    
    def get_proxy_config(self, country_code: str = "US", session_id: str = None) -> Dict[str, str]:
        """Get Decodo proxy configuration with optional targeting"""
        
        username_parts = [f"user-{self.decodo_username}"]
        
        if country_code:
            username_parts.append(f"country-{country_code.lower()}")
        
        if session_id:
            username_parts.append(f"session-{session_id}")
            username_parts.append("sessionduration-30")
        
        proxy_username = "-".join(username_parts)
        proxy_url = f"http://{proxy_username}:{self.decodo_password}@{self.proxy_endpoint}"
        
        return {
            'http': proxy_url,
            'https': proxy_url
        }
    
    def test_proxy_connection(self, country_code: str = "US") -> bool:
        """Test Decodo proxy connection"""
        try:
            proxies = self.get_proxy_config(country_code)
            
            response = requests.get(
                'https://ip.decodo.com/json',
                proxies=proxies,
                timeout=15,
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Decodo proxy working. IP: {data.get('ip', 'unknown')}, Country: {data.get('country', 'unknown')}")
                return True
            else:
                logger.error(f"❌ Proxy test failed with status: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Proxy test error: {str(e)}")
            return False
    
    async def extract_transcript(self, video_id: str, max_retries: int = 3) -> str:
        """Extract YouTube transcript using Decodo residential proxies"""
        logger.info(f"🎯 Starting Decodo extraction for video: {video_id}")
        
        if not self.test_proxy_connection():
            raise Exception("Decodo proxy connection failed. Check your credentials.")
        
        # Try youtube-transcript-api first
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api.proxies import GenericProxyConfig
            
            logger.info("🔧 Using youtube-transcript-api with Decodo proxy...")
            
            session_id = f"yt_{video_id}_{int(time.time())}"
            proxies = self.get_proxy_config("US", session_id)
            
            proxy_config = GenericProxyConfig(
                http_url=proxies['http'],
                https_url=proxies['https']
            )
            
            api = YouTubeTranscriptApi(proxy_config=proxy_config)
            transcript_list = api.list_transcripts(video_id)
            
            transcript = None
            
            try:
                transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB', 'a.en'])
            except:
                available = list(transcript_list)
                if available:
                    transcript = available[0]
                    if transcript.language_code not in ['en', 'en-US', 'en-GB']:
                        transcript = transcript.translate('en')
            
            if not transcript:
                raise Exception("No transcripts available")
            
            transcript_data = transcript.fetch()
            text_parts = []
            
            for entry in transcript_data:
                text = entry['text'].strip()
                if text and not any(noise in text.lower() for noise in [
                    '[music]', '[applause]', '[laughter]', '♪', '♫'
                ]):
                    text_parts.append(text)
            
            if not text_parts:
                raise Exception("No usable transcript content found")
            
            full_transcript = ' '.join(text_parts)
            full_transcript = re.sub(r'\[.*?\]', '', full_transcript)
            full_transcript = re.sub(r'♪.*?♪', '', full_transcript)
            full_transcript = re.sub(r'\s+', ' ', full_transcript).strip()
            
            if len(full_transcript) < 100:
                raise Exception("Transcript too short after cleaning")
            
            logger.info(f"✅ Success with youtube-transcript-api + Decodo: {len(full_transcript)} characters")
            return full_transcript
            
        except ImportError:
            logger.warning("youtube-transcript-api not installed, falling back to custom method")
        except Exception as e:
            logger.warning(f"youtube-transcript-api failed: {str(e)}, trying custom extraction")
        
        # Fallback to custom extraction
        countries_to_try = ["US", "GB", "CA", "AU"]
        
        for country in countries_to_try:
            logger.info(f"🌍 Trying extraction from {country}")
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"🔄 Attempt {attempt + 1}/{max_retries} for {country}")
                    
                    session_id = f"custom_{video_id}_{country}_{attempt}_{int(time.time())}"
                    proxies = self.get_proxy_config(country, session_id)
                    
                    if attempt > 0:
                        delay = random.uniform(2, 4) + attempt
                        await asyncio.sleep(delay)
                    
                    # Try timedtext API
                    transcript = await self._extract_via_timedtext(video_id, proxies)
                    if transcript and len(transcript) > 100:
                        logger.info(f"✅ Success with timedtext from {country}: {len(transcript)} characters")
                        return transcript
                    
                    # Try watch page
                    transcript = await self._extract_via_watch_page(video_id, proxies)
                    if transcript and len(transcript) > 100:
                        logger.info(f"✅ Success with watch page from {country}: {len(transcript)} characters")
                        return transcript
                    
                except Exception as e:
                    logger.warning(f"⚠️ Attempt {attempt + 1} in {country} failed: {str(e)}")
        
        raise Exception(f"All extraction methods failed for video {video_id}")
    
    async def _extract_via_timedtext(self, video_id: str, proxies: Dict[str, str]) -> Optional[str]:
        """Extract via direct timedtext API"""
        urls = [
            f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=json3",
            f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en-US&fmt=json3",
            f"https://www.youtube.com/api/timedtext?v={video_id}&lang=a.en&fmt=json3",
            f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=srv3",
        ]
        
        for url in urls:
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    proxies=proxies,
                    timeout=20
                )
                
                if response.status_code == 200 and response.content:
                    transcript = self._parse_transcript(response.text)
                    if transcript and len(transcript) > 100:
                        return transcript
                        
            except Exception as e:
                logger.debug(f"Failed URL {url}: {e}")
                continue
        
        return None
    
    async def _extract_via_watch_page(self, video_id: str, proxies: Dict[str, str]) -> Optional[str]:
        """Extract from YouTube watch page"""
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
            
            # Find caption URLs
            patterns = [
                r'"captionTracks":\s*(\[[^\]]+\])',
                r'\"baseUrl\":\"(https://www\.youtube\.com/api/timedtext[^\"]*?)\"',
            ]
            
            caption_urls = []
            for pattern in patterns:
                matches = re.findall(pattern, html_content)
                for match in matches:
                    if match.startswith('['):
                        try:
                            tracks = json.loads(match)
                            for track in tracks:
                                if 'baseUrl' in track:
                                    url = track['baseUrl'].replace('\\u0026', '&').replace('\\/', '/')
                                    caption_urls.append(url)
                        except:
                            continue
                    elif 'timedtext' in match:
                        url = match.replace('\\u0026', '&').replace('\\/', '/')
                        caption_urls.append(url)
            
            # Try each caption URL
            for url in caption_urls[:3]:
                try:
                    if not url.startswith('http'):
                        continue
                    
                    if '&fmt=' not in url:
                        url += '&fmt=json3'
                    
                    caption_response = requests.get(
                        url,
                        headers=self.headers,
                        proxies=proxies,
                        timeout=15
                    )
                    
                    if caption_response.status_code == 200:
                        transcript = self._parse_transcript(caption_response.text)
                        if transcript and len(transcript) > 100:
                            return transcript
                            
                except Exception as e:
                    logger.debug(f"Caption URL failed: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.warning(f"Watch page extraction failed: {e}")
            return None
    
    def _parse_transcript(self, content: str) -> Optional[str]:
        """Parse transcript from various formats"""
        try:
            if not content or len(content.strip()) < 10:
                return None
            
            # Try JSON3 format
            if content.strip().startswith('{'):
                try:
                    data = json.loads(content)
                    if 'events' in data:
                        text_parts = []
                        for event in data['events']:
                            if 'segs' in event:
                                for seg in event['segs']:
                                    if 'utf8' in seg:
                                        text = seg['utf8'].strip()
                                        if text and text not in ['[Music]', '[Applause]', '[Laughter]']:
                                            text_parts.append(text)
                        
                        if text_parts:
                            full_transcript = ' '.join(text_parts)
                            full_transcript = re.sub(r'\[.*?\]', '', full_transcript)
                            full_transcript = re.sub(r'\s+', ' ', full_transcript).strip()
                            return full_transcript if len(full_transcript) > 50 else None
                            
                except json.JSONDecodeError:
                    pass
            
            # Try XML format
            if content.strip().startswith('<'):
                try:
                    root = ET.fromstring(content)
                    text_parts = []
                    
                    for elem in root.iter():
                        if elem.tag in ['text', 'p'] and elem.text:
                            text = elem.text.strip()
                            text = re.sub(r'\[.*?\]', '', text)
                            text = re.sub(r'\s+', ' ', text)
                            if text and len(text) > 2:
                                text_parts.append(text)
                    
                    if text_parts:
                        full_transcript = ' '.join(text_parts)
                        full_transcript = re.sub(r'\s+', ' ', full_transcript).strip()
                        return full_transcript if len(full_transcript) > 50 else None
                        
                except ET.ParseError:
                    pass
            
            return None
            
        except Exception as e:
            logger.warning(f"Transcript parsing failed: {e}")
            return None


# Update your global variables
decodo_extractor = None

def initialize_decodo_extractor():
    global decodo_extractor
    
    if not DECODO_USERNAME or not DECODO_PASSWORD:
        raise Exception("Decodo credentials not configured. Set DECODO_USERNAME and DECODO_PASSWORD environment variables")
    
    decodo_extractor = DecodoYouTubeExtractor(DECODO_USERNAME, DECODO_PASSWORD)
    logger.info(f"✅ Decodo YouTube extractor initialized for user: {DECODO_USERNAME}")

# Replace your fetch_transcript_that_actually_works function with this:
async def fetch_transcript_that_actually_works(video_id: str) -> str:
    global decodo_extractor
    
    if not decodo_extractor:
        initialize_decodo_extractor()
    
    logger.info(f"🎬 Starting Decodo extraction for video: {video_id}")
    
    try:
        transcript = await decodo_extractor.extract_transcript(video_id)
        
        if transcript and len(transcript.strip()) > 100:
            logger.info(f"✅ Successfully extracted {len(transcript)} characters with Decodo")
            return transcript
        else:
            raise Exception("Extracted transcript was too short or empty")
            
    except Exception as e:
        logger.error(f"❌ Decodo extractor failed: {str(e)}")
        raise Exception(f"Could not extract transcript with Decodo: {str(e)}")


@app.get("/debug/decodo-test")
async def test_decodo_configuration():
    """Test Decodo proxy configuration"""
    try:
        if not DECODO_USERNAME or not DECODO_PASSWORD:
            return {
                "status": "error",
                "message": "Decodo credentials not configured",
                "fix": "Set DECODO_USERNAME and DECODO_PASSWORD environment variables"
            }
        
        # Test proxy connection
        test_extractor = DecodoYouTubeExtractor(DECODO_USERNAME, DECODO_PASSWORD)
        
        results = {}
        
        # Test different countries
        countries_to_test = ["US", "GB", "CA"]
        
        for country in countries_to_test:
            try:
                # Test basic connectivity
                proxies = test_extractor.get_proxy_config(country)
                
                response = requests.get(
                    'https://ip.decodo.com/json',
                    proxies=proxies,
                    timeout=15,
                    headers=test_extractor.headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Test YouTube accessibility
                    yt_response = requests.get(
                        'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
                        proxies=proxies,
                        timeout=15,
                        headers=test_extractor.headers
                    )
                    
                    results[country] = {
                        "status": "success",
                        "proxy_ip": data.get('ip', 'unknown'),
                        "proxy_country": data.get('country', 'unknown'),
                        "youtube_accessible": yt_response.status_code == 200,
                        "youtube_status": yt_response.status_code
                    }
                else:
                    results[country] = {
                        "status": "failed",
                        "error": f"HTTP {response.status_code}"
                    }
                    
            except Exception as e:
                results[country] = {
                    "status": "error",
                    "error": str(e)
                }
        
        working_countries = [k for k, v in results.items() if v.get("status") == "success" and v.get("youtube_accessible")]
        
        recommendations = []
        if working_countries:
            recommendations.append(f"✅ Working countries: {', '.join(working_countries)}")
            recommendations.append("✅ Decodo residential proxies are working correctly")
        else:
            recommendations.append("❌ No working proxy connections found")
            recommendations.append("💡 Check your Decodo dashboard for active plans")
            recommendations.append("💡 Ensure you have residential proxy credits")
        
        return {
            "status": "diagnostic_complete",
            "decodo_username": DECODO_USERNAME,
            "proxy_test_results": results,
            "recommendations": recommendations,
            "next_steps": [
                "If proxies work, test transcript extraction with: /debug/working-extraction/dQw4w9WgXcQ",
                "Check your Decodo dashboard for usage statistics",
                "Ensure you have residential proxy credits remaining"
            ]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "suggestion": "Check your Decodo credentials and account status"
        }

@app.get("/debug/decodo-extraction/{video_id}")
async def test_decodo_extraction(video_id: str):
    """Test Decodo transcript extraction for a specific video"""
    try:
        logger.info(f"🧪 Testing Decodo extraction for: {video_id}")
        
        transcript = await fetch_transcript_that_actually_works(video_id)
        
        return {
            "status": "success",
            "video_id": video_id,
            "transcript_length": len(transcript),
            "sample": transcript[:300] + "..." if len(transcript) > 300 else transcript,
            "method": "decodo_residential_proxies"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "video_id": video_id,
            "error": str(e),
            "method": "decodo_residential_proxies"
        }

@app.get("/debug/comprehensive-test/{video_id}")
async def comprehensive_youtube_test(video_id: str):
    """Comprehensive test of all YouTube extraction methods"""
    
    results = {
        "video_id": video_id,
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "test_timestamp": time.time(),
        "methods_tested": {}
    }
    
    # Test 1: Check video accessibility
    try:
        response = requests.get(f"https://www.youtube.com/watch?v={video_id}", timeout=10)
        results["video_accessible"] = {
            "status": response.status_code,
            "accessible": response.status_code == 200,
            "content_length": len(response.text) if response.status_code == 200 else 0
        }
    except Exception as e:
        results["video_accessible"] = {
            "status": "error",
            "accessible": False,
            "error": str(e)
        }
    
    # Test 2: YouTube Transcript API without proxy
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        
        api = YouTubeTranscriptApi()
        transcript_list = api.list_transcripts(video_id)
        
        available_transcripts = []
        for transcript in transcript_list:
            available_transcripts.append({
                "language": transcript.language,
                "language_code": transcript.language_code,
                "is_generated": transcript.is_generated,
                "is_translatable": transcript.is_translatable
            })
        
        # Try to get an English transcript
        try:
            transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
            transcript_data = transcript.fetch()
            sample_text = ' '.join([entry['text'] for entry in transcript_data[:3]])
            
            results["methods_tested"]["youtube_transcript_api_direct"] = {
                "status": "success",
                "available_transcripts": available_transcripts,
                "extracted_length": len(transcript_data),
                "sample": sample_text[:200] + "..." if len(sample_text) > 200 else sample_text
            }
        except Exception as inner_e:
            results["methods_tested"]["youtube_transcript_api_direct"] = {
                "status": "partial_success",
                "available_transcripts": available_transcripts,
                "extraction_error": str(inner_e)
            }
            
    except ImportError:
        results["methods_tested"]["youtube_transcript_api_direct"] = {
            "status": "not_available",
            "error": "youtube-transcript-api not installed"
        }
    except Exception as e:
        results["methods_tested"]["youtube_transcript_api_direct"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Test 3: YouTube Transcript API with Decodo proxy
    if DECODO_USERNAME and DECODO_PASSWORD:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api.proxies import GenericProxyConfig
            
            # Test proxy first
            test_extractor = UltimateYouTubeExtractor(DECODO_USERNAME, DECODO_PASSWORD)
            proxies = test_extractor.get_proxy_config("US", f"test_{int(time.time())}")
            
            proxy_test = requests.get('https://ip.decodo.com/json', proxies=proxies, timeout=10)
            
            if proxy_test.status_code == 200:
                proxy_info = proxy_test.json()
                
                proxy_config = GenericProxyConfig(
                    http_url=proxies['http'],
                    https_url=proxies['https']
                )
                
                api = YouTubeTranscriptApi(proxy_config=proxy_config)
                transcript_list = api.list_transcripts(video_id)
                
                try:
                    transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
                    transcript_data = transcript.fetch()
                    sample_text = ' '.join([entry['text'] for entry in transcript_data[:3]])
                    
                    results["methods_tested"]["youtube_transcript_api_with_proxy"] = {
                        "status": "success",
                        "proxy_ip": proxy_info.get('ip', 'unknown'),
                        "proxy_country": proxy_info.get('country', 'unknown'),
                        "extracted_length": len(transcript_data),
                        "sample": sample_text[:200] + "..." if len(sample_text) > 200 else sample_text
                    }
                except Exception as inner_e:
                    results["methods_tested"]["youtube_transcript_api_with_proxy"] = {
                        "status": "proxy_works_but_extraction_failed",
                        "proxy_ip": proxy_info.get('ip', 'unknown'),
                        "proxy_country": proxy_info.get('country', 'unknown'),
                        "extraction_error": str(inner_e)
                    }
            else:
                results["methods_tested"]["youtube_transcript_api_with_proxy"] = {
                    "status": "proxy_test_failed",
                    "proxy_status": proxy_test.status_code
                }
                
        except Exception as e:
            results["methods_tested"]["youtube_transcript_api_with_proxy"] = {
                "status": "error",
                "error": str(e)
            }
    else:
        results["methods_tested"]["youtube_transcript_api_with_proxy"] = {
            "status": "not_configured",
            "error": "Decodo credentials not provided"
        }
    
    # Test 4: Direct timedtext API calls
    timedtext_urls = [
        f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=json3",
        f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en-US&fmt=json3",
        f"https://www.youtube.com/api/timedtext?v={video_id}&lang=a.en&fmt=json3",
        f"https://video.google.com/timedtext?lang=en&v={video_id}",
    ]
    
    timedtext_results = []
    for url in timedtext_urls:
        try:
            response = requests.get(url, timeout=10)
            timedtext_results.append({
                "url": url,
                "status": response.status_code,
                "content_length": len(response.text),
                "has_content": len(response.text) > 50,
                "sample": response.text[:100] + "..." if len(response.text) > 100 else response.text
            })
        except Exception as e:
            timedtext_results.append({
                "url": url,
                "status": "error",
                "error": str(e)
            })
    
    results["methods_tested"]["direct_timedtext"] = {
        "status": "tested",
        "results": timedtext_results
    }
    
    # Test 5: YouTube oembed (for video info)
    try:
        oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        response = requests.get(oembed_url, timeout=10)
        if response.status_code == 200:
            video_info = response.json()
            results["video_info"] = {
                "title": video_info.get('title', 'Unknown'),
                "author": video_info.get('author_name', 'Unknown'),
                "duration": video_info.get('duration', 'Unknown')
            }
        else:
            results["video_info"] = {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        results["video_info"] = {"error": str(e)}
    
    # Test 6: Check if video has captions by looking at watch page
    try:
        watch_response = requests.get(f"https://www.youtube.com/watch?v={video_id}", timeout=15)
        if watch_response.status_code == 200:
            html_content = watch_response.text
            
            # Look for caption indicators
            caption_indicators = {
                "has_captions_button": '"captions"' in html_content or '"cc"' in html_content,
                "has_transcript_menu": '"transcript"' in html_content.lower(),
                "has_timedtext_references": 'timedtext' in html_content,
                "has_caption_tracks": '"captionTracks"' in html_content,
                "page_title_in_html": video_info.get('title', 'UNKNOWN') in html_content if 'video_info' in results else False
            }
            
            # Try to extract any caption URLs from the page
            caption_url_patterns = [
                r'"captionTracks":\s*(\[[^\]]+\])',
                r'\"baseUrl\":\"(https://[^\"]*?timedtext[^\"]*?)\"'
            ]
            
            found_caption_urls = []
            for pattern in caption_url_patterns:
                matches = re.findall(pattern, html_content)
                for match in matches:
                    if match.startswith('['):
                        try:
                            tracks = json.loads(match)
                            for track in tracks:
                                if isinstance(track, dict) and 'baseUrl' in track:
                                    url = track['baseUrl'].replace('\\u0026', '&').replace('\\/', '/')
                                    found_caption_urls.append(url)
                        except:
                            continue
                    elif 'timedtext' in match:
                        url = match.replace('\\u0026', '&').replace('\\/', '/')
                        found_caption_urls.append(url)
            
            results["methods_tested"]["watch_page_analysis"] = {
                "status": "success",
                "caption_indicators": caption_indicators,
                "found_caption_urls": found_caption_urls[:3],  # Limit to first 3
                "total_caption_urls_found": len(found_caption_urls)
            }
        else:
            results["methods_tested"]["watch_page_analysis"] = {
                "status": "failed",
                "http_status": watch_response.status_code
            }
    except Exception as e:
        results["methods_tested"]["watch_page_analysis"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Test 7: Ultimate extractor test
    try:
        ultimate_extractor = UltimateYouTubeExtractor(DECODO_USERNAME, DECODO_PASSWORD)
        transcript = await ultimate_extractor.extract_transcript(video_id)
        
        results["methods_tested"]["ultimate_extractor"] = {
            "status": "success",
            "transcript_length": len(transcript),
            "sample": transcript[:300] + "..." if len(transcript) > 300 else transcript
        }
    except Exception as e:
        results["methods_tested"]["ultimate_extractor"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Generate recommendations based on test results
    recommendations = []
    
    if not results["video_accessible"]["accessible"]:
        recommendations.append("❌ Video is not accessible - check if it exists and is public")
    
    if results.get("video_info", {}).get("error"):
        recommendations.append("⚠️ Could not get video info - video might be private or deleted")
    
    # Check if any method worked
    successful_methods = []
    for method, result in results["methods_tested"].items():
        if result.get("status") == "success":
            successful_methods.append(method)
    
    if successful_methods:
        recommendations.append(f"✅ Working methods: {', '.join(successful_methods)}")
    else:
        recommendations.append("❌ No extraction methods worked")
        
        # Specific troubleshooting
        if any(result.get("status") == "not_configured" for result in results["methods_tested"].values()):
            recommendations.append("💡 Configure Decodo proxy credentials for better success rate")
        
        watch_page = results["methods_tested"].get("watch_page_analysis", {})
        if watch_page.get("status") == "success":
            indicators = watch_page.get("caption_indicators", {})
            if not any(indicators.values()):
                recommendations.append("💡 Video may not have captions available")
            elif indicators.get("has_caption_tracks") and watch_page.get("total_caption_urls_found", 0) == 0:
                recommendations.append("💡 Video has captions but URLs couldn't be extracted (possible new protection)")
        
        # Check timedtext results
        timedtext = results["methods_tested"].get("direct_timedtext", {})
        if timedtext.get("status") == "tested":
            working_timedtext = [r for r in timedtext.get("results", []) if r.get("status") == 200 and r.get("has_content")]
            if working_timedtext:
                recommendations.append("💡 Direct timedtext API works - issue may be with extraction logic")
            else:
                recommendations.append("💡 Direct timedtext API blocked - YouTube has likely enhanced protection")
    
    results["recommendations"] = recommendations
    results["summary"] = {
        "total_methods_tested": len(results["methods_tested"]),
        "successful_methods": len(successful_methods),
        "video_likely_has_captions": (
            results["methods_tested"].get("watch_page_analysis", {})
            .get("caption_indicators", {})
            .get("has_caption_tracks", False)
        ),
        "proxy_working": (
            results["methods_tested"].get("youtube_transcript_api_with_proxy", {})
            .get("status") in ["success", "proxy_works_but_extraction_failed"]
        )
    }
    
    return results


@app.get("/debug/check-video-captions/{video_id}")
async def check_video_captions_manually(video_id: str):
    """Simple check to see if a video has captions available"""
    try:
        # Get video page
        response = requests.get(f"https://www.youtube.com/watch?v={video_id}", timeout=15)
        
        if response.status_code != 200:
            return {
                "status": "error",
                "message": f"Could not access video page (HTTP {response.status_code})"
            }
        
        html_content = response.text
        
        # Check for various caption indicators
        indicators = {
            "has_closed_captions": '"closedCaptionsRenderer"' in html_content,
            "has_caption_tracks": '"captionTracks"' in html_content,
            "has_subtitle_tracks": '"subtitleTracks"' in html_content,
            "has_auto_captions": '"kind":"asr"' in html_content,
            "has_manual_captions": '"kind":"caption"' in html_content,
        }
        
        # Try to extract specific language information
        languages_found = []
        lang_pattern = r'"languageCode":"([^"]+)"'
        lang_matches = re.findall(lang_pattern, html_content)
        languages_found = list(set(lang_matches))
        
        has_captions = any(indicators.values()) or len(languages_found) > 0
        
        return {
            "status": "success",
            "video_id": video_id,
            "has_captions": has_captions,
            "caption_indicators": indicators,
            "languages_found": languages_found,
            "manual_check_url": f"https://www.youtube.com/watch?v={video_id}",
            "instructions": "Open the manual_check_url and look for the CC button below the video to confirm captions are available"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/debug/test-working-videos")
async def test_known_working_videos():
    """Test extraction on videos that are known to have captions"""
    
    # Known videos with captions
    test_videos = [
        {
            "id": "dQw4w9WgXcQ", 
            "title": "Rick Astley - Never Gonna Give You Up",
            "expected_captions": True
        },
        {
            "id": "jNQXAC9IVRw", 
            "title": "Me at the zoo (first YouTube video)",
            "expected_captions": True
        },
        {
            "id": "9bZkp7q19f0", 
            "title": "PSY - Gangnam Style",
            "expected_captions": True
        }
    ]
    
    results = []
    
    for video in test_videos:
        try:
            # Quick test with ultimate extractor
            if 'ultimate_extractor' not in globals():
                ultimate_extractor = UltimateYouTubeExtractor(DECODO_USERNAME, DECODO_PASSWORD)
            
            transcript = await ultimate_extractor.extract_transcript(video["id"])
            
            results.append({
                "video_id": video["id"],
                "title": video["title"],
                "status": "success",
                "transcript_length": len(transcript),
                "sample": transcript[:150] + "..." if len(transcript) > 150 else transcript
            })
            
        except Exception as e:
            results.append({
                "video_id": video["id"],
                "title": video["title"],
                "status": "failed",
                "error": str(e)
            })
    
    successful_extractions = len([r for r in results if r["status"] == "success"])
    
    return {
        "total_videos_tested": len(test_videos),
        "successful_extractions": successful_extractions,
        "success_rate": f"{(successful_extractions / len(test_videos)) * 100:.1f}%",
        "results": results,
        "interpretation": {
            "all_success": "Extraction is working correctly" if successful_extractions == len(test_videos) else None,
            "partial_success": f"Extraction works for some videos ({successful_extractions}/{len(test_videos)})" if 0 < successful_extractions < len(test_videos) else None,
            "no_success": "Extraction is completely blocked or misconfigured" if successful_extractions == 0 else None
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)