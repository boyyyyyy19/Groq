import subprocess
import sys
import os
import re
from datetime import datetime
import time
import json
from typing import List, Dict, Set, Optional
from urllib.parse import urlparse, parse_qs
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import warnings
import logging

# Suppress all warnings
warnings.filterwarnings('ignore')

# Configure logging to only show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Create a filter to only show INFO messages
class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO

# Add filter to the root logger
logging.getLogger().addFilter(InfoFilter())

class TokenBucket:
    """
    Implements rate limiting using the token bucket algorithm.
    """
    def __init__(self, tokens: int, fill_rate: float):
        self.capacity = tokens
        self.tokens = tokens
        self.fill_rate = fill_rate
        self.last_update = time.time()

    def consume(self, tokens: int = 1) -> bool:
        now = time.time()
        self.tokens += (now - self.last_update) * self.fill_rate
        self.tokens = min(self.tokens, self.capacity)
        self.last_update = now

        self.tokens -= tokens
        return self.tokens >= 0

def install_required_packages():
    """Check and install required packages if they're missing."""
    required_packages = [
        'groq',
        'requests',
        'beautifulsoup4',
        'urllib3',
        'tqdm',
        'duckduckgo_search',
        'youtube-transcript-api',
        'yt-dlp'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print("Installing modules")  # Added this line to show installation message
            subprocess.check_call([sys.executable, "-m", "pip", "install", package],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)

# [Rest of the classes and code remains the same...]

class Crawl4AI:
    """
    Web crawler optimized for AI context gathering.
    """
    def __init__(self, max_pages: int = 5, rate_limit: int = 1):
        self.max_pages = max_pages
        self.rate_limiter = TokenBucket(tokens=rate_limit, fill_rate=0.5)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; Crawl4AI/1.0; +http://example.com/bot)'
        })
        # Disable SSL verification warnings
        requests.packages.urllib3.disable_warnings()

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text

    def extract_text(self, soup: BeautifulSoup) -> str:
        """Extract meaningful text from HTML content."""
        for element in soup(['script', 'style', 'header', 'footer', 'nav']):
            element.decompose()
        return self.clean_text(soup.get_text(separator=' ', strip=True))

    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and allowed."""
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False

    def crawl(self, url: str) -> List[Dict]:
        """Crawl a webpage and extract relevant content."""
        if not self.is_valid_url(url):
            return []

        results = []
        try:
            while not self.rate_limiter.consume():
                time.sleep(0.1)

            response = self.session.get(url, timeout=10, verify=False)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            title = soup.title.string if soup.title else ''
            description = soup.find('meta', attrs={'name': 'description'})
            description = description.get('content', '') if description else ''
            text = self.extract_text(soup)

            results.append({
                'url': url,
                'title': self.clean_text(title),
                'description': self.clean_text(description),
                'text': text,
                'timestamp': datetime.now().isoformat()
            })

        except Exception:
            pass

        return results

class YouTubeSummarizer:
    """Handles YouTube video summarization functionality."""
    def __init__(self):
        import yt_dlp
        self.yt_dlp = yt_dlp
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'no_color': True
        }

    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from various URL formats."""
        try:
            parsed_url = urlparse(url)
            if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
                if parsed_url.path == '/watch':
                    return parse_qs(parsed_url.query)['v'][0]
                elif parsed_url.path.startswith('/shorts/'):
                    return parsed_url.path.split('/')[2]
            elif parsed_url.hostname == 'youtu.be':
                return parsed_url.path[1:]
        except Exception:
            pass
        return None

    def get_video_metadata(self, url: str) -> Dict:
        """Get video metadata using yt-dlp."""
        try:
            with self.yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'view_count': info.get('view_count', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'upload_date': info.get('upload_date', 'Unknown')
                }
        except Exception:
            return {}

    def get_transcript(self, video_id: str) -> Optional[str]:
        """Get video transcript using youtube-transcript-api."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api.formatters import TextFormatter
            
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            formatter = TextFormatter()
            return formatter.format_transcript(transcript)
        except Exception:
            return None

    def summarize(self, url: str) -> Dict:
        """Get comprehensive summary of a YouTube video."""
        video_id = self.extract_video_id(url)
        if not video_id:
            return {"error": "Invalid YouTube URL"}

        metadata = self.get_video_metadata(url)
        transcript = self.get_transcript(video_id)

        return {
            "video_id": video_id,
            "metadata": metadata,
            "transcript": transcript,
            "url": url
        }

class WebContextFinder:
    """Finds and analyzes web context including YouTube videos."""
    def __init__(self, max_search_results: int = 5):
        self.max_search_results = max_search_results
        self.youtube_summarizer = YouTubeSummarizer()

    def is_youtube_url(self, url: str) -> bool:
        """Check if the given URL is a YouTube URL."""
        parsed_url = urlparse(url)
        return any(domain in parsed_url.netloc 
                  for domain in ['youtube.com', 'youtu.be', 'www.youtube.com'])

    def find_relevant_context(self, query: str) -> str:
        """Find relevant web context, handling YouTube URLs specially."""
        try:
            if self.is_youtube_url(query):
                video_info = self.youtube_summarizer.summarize(query)
                
                if "error" in video_info:
                    return "Unable to process YouTube video."
                
                metadata = video_info["metadata"]
                if not metadata:
                    return "Unable to retrieve video information."
                
                context_parts = [
                    f"YouTube Video Summary:",
                    f"Title: {metadata.get('title')}",
                    f"Uploader: {metadata.get('uploader')}",
                    f"Upload Date: {metadata.get('upload_date')}",
                    f"Duration: {metadata.get('duration')} seconds",
                    f"Views: {metadata.get('view_count')}",
                ]
                
                if video_info.get("transcript"):
                    context_parts.extend([
                        "\nTranscript Summary:",
                        video_info["transcript"][:1000] + "..."
                    ])
                
                return "\n".join(context_parts)
            
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                search_results = list(ddgs.text(query, max_results=self.max_search_results))
            
            context_parts = []
            crawler = Crawl4AI(max_pages=1)
            
            for result in search_results:
                try:
                    url = result.get('href', '')
                    if self.is_youtube_url(url):
                        video_info = self.youtube_summarizer.summarize(url)
                        if "error" not in video_info and video_info['metadata']:
                            context_parts.append(f"Related YouTube Video: {video_info['metadata']['title']}")
                    else:
                        crawl_result = crawler.crawl(url)
                        if crawl_result:
                            context_parts.append(f"Source: {url}")
                            text = crawl_result[0].get('text', '')[:500]
                            context_parts.append(f"Content Summary: {text}\n")
                except Exception:
                    continue
            
            return "\n".join(context_parts) if context_parts else "No relevant context found."
            
        except Exception:
            return "Unable to find web context."

def determine_need_for_web_context(question: str) -> bool:
    """Determine if the question requires additional web context."""
    if any(domain in question.lower() 
           for domain in ['youtube.com', 'youtu.be']):
        return True
        
    context_keywords = [
        'recent', 'latest', 'current', 'today', 'now', 
        'happening', 'update', 'status', 'news', 
        'who is', 'what is', 'when did', 'how do', 
        'explain', 'define', 'describe', 'video'
    ]
    
    question = question.lower()
    return any(keyword in question for keyword in context_keywords) or len(question.split()) > 5

def main():
    """Main function to run the web context and analysis tool."""
    try:
        # Install required packages
        install_required_packages()

        from groq import Groq
        
        api_key = "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"
        if not api_key:
            print("API key not set. Please set your Groq API key.")
            return
            
        client = Groq(api_key=api_key)

        print("\nWeb Context and Analysis Tool")
        print("=" * 30)
        print("1. Ask any question")
        print("2. Paste a YouTube URL for summary")
        print("3. Type 'quit' to exit")
        print("=" * 30)
        
        while True:
            content = input("\nEnter your question, YouTube URL, or prompt (or 'quit' to exit): \n").strip()
            
            if content.lower() in ['quit', 'exit', 'bye']:
                print("\nThank you for using the tool. Goodbye!")
                break

            if not content:
                print("Please enter a valid input.")
                continue

            need_web_context = determine_need_for_web_context(content)
            crawled_content = ""
            
            if need_web_context:
                print("\nAnalyzing content...")
                web_context_finder = WebContextFinder()
                crawled_content = web_context_finder.find_relevant_context(content)
                print("\nAnalysis complete. Generating response...\n")

            messages = [
                {
                    "role": "system",
                    "content": """You are an AI assistant designed to help users with general questions 
                    and YouTube video summaries. Always use chain of thought reasoning and provide clear, 
                    concise responses. For YouTube videos, focus on key points and main takeaways."""
                }
            ]
            
            if crawled_content:
                messages.append({
                    "role": "system",
                    "content": f"Content Analysis Results: {crawled_content}"
                })
            
            messages.append({
                "role": "user",
                "content": content,
            })

            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    chat_completion = client.chat.completions.create(
                        messages=messages,
                        model="llama-3.2-90b-vision-preview"
                    )
                    
                    ai_response = chat_completion.choices[0].message.content
                    print("\nAI Response:")
                    print(ai_response)
                    break
                    
                except Exception:
                    if attempt == max_retries - 1:
                        print("Unable to generate response. Please try again.")
                    time.sleep(retry_delay * (attempt + 1))

    except Exception as e:
        print("An error occurred. Please try again.")

if __name__ == "__main__":
    main()
