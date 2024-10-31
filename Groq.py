import subprocess
import sys
import os
import re
from datetime import datetime
import time
import logging
import json
from typing import List, Dict, Set, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def install_required_packages():
    """
    Check and install required packages if they're missing.
    """
    required_packages = [
        'groq',
        'requests',
        'beautifulsoup4',
        'urllib3',
        'tqdm',
        'duckduckgo_search'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing required package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
print("Checking and installing required packages...")
install_required_packages()

# Now import the required packages
from groq import Groq
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm
from duckduckgo_search import DDGS

class TokenBucket:
    """
    Implements token bucket algorithm for rate limiting.
    """
    def __init__(self, tokens_per_second: float, bucket_size: int):
        self.tokens_per_second = tokens_per_second
        self.bucket_size = bucket_size
        self.tokens = bucket_size
        self.last_update = time.time()

    def get_token(self):
        now = time.time()
        time_passed = now - self.last_update
        self.tokens = min(
            self.bucket_size,
            self.tokens + time_passed * self.tokens_per_second
        )
        self.last_update = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    def wait_for_token(self):
        while not self.get_token():
            time.sleep(0.1)

class Crawl4AI:
    def __init__(self, max_pages: int = 10, delay: float = 1.0):
        """
        Initialize the web crawler with rate limiting.
        """
        self.max_pages = max_pages
        self.delay = delay
        self.visited_urls: Set[str] = set()
        self.results: List[Dict] = []
        self.token_bucket = TokenBucket(tokens_per_second=2, bucket_size=10)
        
        # Configure headers for polite crawling
        self.headers = {
            'User-Agent': 'Crawl4AI Bot 1.0 - Educational Purpose Web Crawler',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        # Setup session with retry mechanism
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def is_valid_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc) and bool(parsed.scheme)
        except:
            return False

    def extract_text_and_links(self, url: str) -> Optional[Dict]:
        """
        Extract text content and links from a webpage with rate limiting.
        """
        try:
            # Wait for rate limit token
            self.token_bucket.wait_for_token()
            
            # Add delay for politeness
            time.sleep(self.delay)
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for element in soup(["script", "style", "meta", "link"]):
                element.decompose()
            
            # Extract text
            text = soup.get_text(separator='\n', strip=True)
            
            # Extract links
            links = []
            for link in soup.find_all('a'):
                href = link.get('href')
                if href:
                    absolute_url = urljoin(url, href)
                    if self.is_valid_url(absolute_url):
                        links.append(absolute_url)
            
            return {
                'url': url,
                'text': text,
                'links': links,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error crawling {url}: {str(e)}")
            return None

    def crawl(self, start_url: str) -> List[Dict]:
        """
        Start crawling from a given URL with progress bar.
        """
        if not self.is_valid_url(start_url):
            raise ValueError("Invalid start URL provided")
        
        urls_to_visit = [start_url]
        pbar = tqdm(total=self.max_pages, desc="Crawling pages")
        
        while urls_to_visit and len(self.visited_urls) < self.max_pages:
            current_url = urls_to_visit.pop(0)
            
            if current_url in self.visited_urls:
                continue
                
            logging.info(f"Crawling: {current_url}")
            self.visited_urls.add(current_url)
            
            result = self.extract_text_and_links(current_url)
            if result:
                self.results.append(result)
                urls_to_visit.extend([url for url in result['links'] 
                                   if url not in self.visited_urls])
                pbar.update(1)
        
        pbar.close()
        return self.results

    def save_results(self, filename: str = "crawl_results.json"):
        """
        Save crawl results to a JSON file.
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

class WebContextFinder:
    def __init__(self, max_search_results: int = 5):
        """
        Initialize web context finder with DuckDuckGo search.
        """
        self.max_search_results = max_search_results

    def find_relevant_context(self, query: str) -> str:
        """
        Find relevant web context for a given query using DuckDuckGo search.
        """
        try:
            # Use DuckDuckGo search
            with DDGS() as ddgs:
                search_results = list(ddgs.text(query, max_results=self.max_search_results))
            
            # Combine search results into context
            context_parts = []
            crawler = Crawl4AI(max_pages=1)  # Limit to 1 page per result
            
            for result in search_results:
                try:
                    url = result.get('href', '')
                    crawl_result = crawler.crawl(url)
                    
                    # Combine title and snippet if available
                    context_parts.append(f"Source: {url}\n")
                    
                    if crawl_result and len(crawl_result) > 0:
                        # Take first 500 characters of text
                        text = crawl_result[0].get('text', '')[:500]
                        context_parts.append(f"Content Summary: {text}\n")
                    
                except Exception as e:
                    logging.error(f"Error processing search result {url}: {str(e)}")
            
            return "\n".join(context_parts)
        
        except Exception as e:
            logging.error(f"Error finding web context: {str(e)}")
            return f"Could not find web context. Error: {str(e)}"

def determine_need_for_web_context(question: str) -> bool:
    """
    Determine if the question requires additional web context.
    
    This function uses some heuristics to decide if web searching would be helpful:
    - Check for knowledge-seeking questions
    - Identify topics that might benefit from current information
    """
    # Keywords that suggest need for web context
    context_keywords = [
        'recent', 'latest', 'current', 'today', 'now', 
        'happening', 'update', 'status', 'news', 
        'who is', 'what is', 'when did', 'how do', 
        'explain', 'define', 'describe'
    ]
    
    # Convert question to lowercase for case-insensitive matching
    lower_question = question.lower()
    
    # Check if any context keywords are present
    for keyword in context_keywords:
        if keyword in lower_question:
            return True
    
    # Additional length and complexity check
    if len(lower_question.split()) > 5:
        return True
    
    return False

def main():
    try:
        # Initialize Groq client
        client = Groq(
            api_key="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"  # Replace with your actual Groq API key
        )

        print("Hello there! I can help you with questions and web content analysis.\n")
        
        # Main interaction loop
        while True:
            # Get user's question
            content = input("\nEnter your question or prompt (or 'Quit' to exit): \n")
            
            # Check for exit commands
            if content.lower() in ['quit', 'exit', 'bye']:
                print("\nThank you for using the program. Goodbye!")
                break

            # Determine if web context is needed
            need_web_context = determine_need_for_web_context(content)
            crawled_content = ""
            
            if need_web_context:
                print("\nFinding relevant web context...")
                web_context_finder = WebContextFinder()
                crawled_content = web_context_finder.find_relevant_context(content)
                print(f"\nFound web context. Enhancing AI response...\n")

            # Prepare messages for AI
            messages = [
                {
                    "role": "system",
                    "content": '''You are an AI assistant designed to help the user. Use chain of thought reasoning. 
                    Use bullet / numbered lists IF AND ONLY IF you need to. 
                    When sharing code examples, always use markdown code blocks with appropriate language specification.
                    If web context is provided, incorporate it into your response thoughtfully.'''
                }
            ]
            
            # Add crawled content as context if available
            if crawled_content:
                messages.append({
                    "role": "system",
                    "content": f"Supplementary Web Context: {crawled_content}"
                })
            
            # Add user's question
            messages.append({
                "role": "user",
                "content": content,
            })

            # Retry mechanism for API calls
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    chat_completion = client.chat.completions.create(
                        messages=messages,
                        model="llama-3.2-90b-vision-preview"
                    )
                    
                    # Process the response
                    AI_response = chat_completion.choices[0].message.content
                    print("\nAI Response:")
                    print(AI_response)
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        logging.error(f"Attempt {attempt + 1} failed: {str(e)}")
                        time.sleep(retry_delay * (attempt + 1))
                    else:
                        print(f"Error: Failed to get response after {max_retries} attempts.")
                        logging.error(f"Final attempt failed: {str(e)}")

    except Exception as e:
        logging.error(f"Main function error: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
