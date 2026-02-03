import logging
import os
import queue
import threading
import time
import json
from concurrent.futures import Future, ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import requests
from openai import OpenAI


JINA_API_KEY = os.environ.get("JINA_API_KEY")
from rllm.tools.tool_base import Tool, ToolOutput

logger = logging.getLogger(__name__)


def search_with_jina(query):
    query = query.replace(' ', '+')
    url = f"https://s.jina.ai/?q={query}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
        "X-Engine": "direct"
    }

    retries = 3
    for _ in range(retries):
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()['data']
    return None

SUMMARY_PROMPT = """You are an AI research assistant. Your task is to analyze the provided webpage content and extract information relevant to the user's query.

Search Query: {search_query}
Webpage Content:
{first_page_fetch_res}

Please analyze the content and return a **valid JSON object** with the following fields:
- "is_helpful": (boolean) Whether the content contains information relevant to the user's query.
- "summary": (string) A concise, information-dense summary of the content relevant to the query. If 'is_helpful' is false, briefly explain why (e.g., "Content is behind a paywall" or "Topic unrelated").
- "relevance_score": (integer) A score from 0-10 indicating how relevant this page is.

Ensure the output is raw JSON, without Markdown formatting (like ```json).
"""

def extract_info_with_llm(
    query: str, 
    search_results: List[Dict], 
    client: Any, 
    model_name: str = "gpt-4o"
) -> List[Dict]:
    extracted_infos = []

    for result in search_results:
        link = result.get('url')
        title = result.get('title')
        
        print(f"Processing: {title} ({link})")
        
        page_content = result.get('content')
        if not page_content or page_content.startswith("Error:") or len(page_content) < 50:
            print(f"Skipping {link} due to insufficient content.")
            continue
            
        truncated_content = page_content[:15000] 
        
        prompt_content = SUMMARY_PROMPT.format(
            search_query=query, 
            first_page_fetch_res=truncated_content
        )

        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt_content},
        ]

        try:
            kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.6
            }
            
            try:
                response = client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
            except TypeError:
                response = client.chat.completions.create(**kwargs)

            llm_output = response.choices[0].message.content
            summary = ""
            is_helpful = False
            
            try:
                clean_json_str = llm_output.replace("```json", "").replace("```", "").strip()
                
                data = json.loads(clean_json_str)
                summary = data.get("summary", "")
                is_helpful = data.get("is_helpful", False)

                if not is_helpful:
                    print(f"  - Webpage marked as not helpful: {summary[:50]}...")

            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing failed for {link}: {e}. Raw output: {llm_output[:100]}...")
                summary = llm_output

            extracted_infos.append({
                "title": title,
                "url": link,
                "summary": summary,
                "is_helpful": is_helpful, 
                "full_llm_response": llm_output
            })
            
        except Exception as e:
            logging.error(f"LLM processing failed for {link}: {e}")
            continue

    return extracted_infos

@dataclass
class _BatchTask:
    query: str
    top_k: int
    future: Future = field(default_factory=Future)


class _CacheManager:
    """
    Manages local JSON cache with thread-safe read/write operations.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.lock = threading.Lock()
        self.data: Dict[str, Any] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        """Loads the cache file into memory on initialization."""
        if not os.path.exists(self.file_path):
            logger.info(f"Cache file {self.file_path} not found, creating new one.")
            self.data = {}
            return

        try:
            with self.lock:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if not content.strip():
                        self.data = {}
                    else:
                        self.data = json.loads(content)
            logger.info(f"Loaded {len(self.data)} entries from cache file: {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to load cache file {self.file_path}: {e}")
            self.data = {}

    def get(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Returns cached results for a query if exists."""
        with self.lock:
            return self.data.get(query)

    def set(self, query: str, results: List[Dict[str, Any]]):
        """Updates memory and writes to disk immediately."""
        if not results:
            return # Don't cache empty or failed results

        with self.lock:
            self.data[query] = results
            try:
                # Write to a temporary file first to avoid corruption
                temp_path = f"{self.file_path}.tmp"
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=2)
                os.replace(temp_path, self.file_path)
                logger.debug(f"Cached query: '{query}' to {self.file_path}")
            except Exception as e:
                logger.error(f"Failed to save cache to {self.file_path}: {e}")


exclude_url = ['huggingface', 'gaia']

class _WebBatchRequester:
    """
    Aggregates web search queries across tool instances and executes them in parallel batches.
    Includes a caching layer.
    """

    def __init__(
        self,
        timeout: float,
        cache_file_path: str, # Added cache path
        max_batch_size: int = 1,
        batch_interval: float = 0.01,
    ):
        llm_api_key =  os.environ.get("OPENAI_API_KEY")
        llm_base_url =  os.environ.get("OPENAI_BASE_URL")
        self.llm_model = os.environ.get("SUMMARY_MODEL")
        self.llm_client = OpenAI(api_key=llm_api_key, base_url=llm_base_url)
        self.timeout = timeout
        self.max_batch_size = max_batch_size
        self.batch_interval = batch_interval
        self.queue: queue.Queue[_BatchTask] = queue.Queue()
        
        # Initialize Cache Manager
        self.cache = _CacheManager(cache_file_path)
        
        # Identifier for the thread
        self._id = f"web-search-{id(self)}"
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker,
            name=f"retrieval-batcher-{self._id}",
            daemon=True,
        )
        self._worker_thread.start()

    def submit(self, query: str, top_k: int) -> Future:
        # 1. Check Cache First
        cached_results = self.cache.get(query)
        if cached_results is not None:
            logger.info(f"Cache HIT for query: '{query}'")
            # If hit, return a completed future immediately
            f = Future()
            f.set_result(cached_results)
            return f

        # 2. If Miss, Add to Queue
        logger.info(f"Cache MISS for query: '{query}', queuing task.")
        task = _BatchTask(query=query, top_k=top_k)
        self.queue.put(task)
        return task.future

    def close(self):
        self._stop_event.set()
        self._worker_thread.join(timeout=1.0)

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                task = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            batch: List[_BatchTask] = [task]
            deadline = time.perf_counter() + self.batch_interval

            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.queue.get(timeout=remaining))
                except queue.Empty:
                    break

            try:
                self._process_batch(batch)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Batch web retrieval failed: %s", exc)
                for task_item in batch:
                    if not task_item.future.done():
                        task_item.future.set_exception(exc)
            finally:
                for _ in batch:
                    self.queue.task_done()

    def _execute_single_search(self, task: _BatchTask) -> List[Dict[str, Any]]:
        """
        Executes the full web search pipeline for a single query:
        Search -> Scrape -> LLM Summarize -> **Save to Cache**
        """
        try:
            # 1. Search API
            raw_results = search_with_jina(query=task.query)
            filtered_results = []
            for ret in raw_results:
                filtered = False
                for url in exclude_url:
                    if url in ret.get('url', " ").lower():
                        filtered = True
                        break
                if not filtered:
                    filtered_results.append(ret)
            raw_results = filtered_results

            # 2. Extract & Summarize
            extracted_infos = extract_info_with_llm(
                query=task.query,
                search_results=raw_results,
                client=self.llm_client,
                model_name=self.llm_model
            )
            
            # 3. Format result
            docs = []
            for info in extracted_infos:
                docs.append({
                    "id": info.get("url", "unknown"),
                    "score": 1.0, 
                    "content": f"Title: {info.get('title')}\nURL: {info.get('url')}\nSummary: {info.get('summary')}",
                    "raw_info": info 
                })
            
            # 4. Save to Cache dynamically
            if docs:
                self.cache.set(task.query, docs)
            
            return docs

        except Exception as e:
            logger.error(f"Error processing query '{task.query}': {e}")
            return []

    def _process_batch(self, batch: List[_BatchTask]):
        """
        Process the batch concurrently using a ThreadPool.
        """
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            future_to_task = {
                executor.submit(self._execute_single_search, task): task 
                for task in batch
            }

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    results = future.result()
                    if not task.future.done():
                        task.future.set_result(results)
                except Exception as exc:
                    if not task.future.done():
                        task.future.set_exception(exc)



class WebRetrievalTool(Tool):
    """
    A tool for web search that batches queries, processes them asynchronously, 
    and caches results locally.
    """

    NAME = "web_search"
    DESCRIPTION = "Search for information on the internet. Useful for up-to-date events or specific external knowledge."

    _aggregators: Dict[str, _WebBatchRequester] = {}
    _aggregators_lock = threading.Lock()

    def __init__(
        self,
        name: str = NAME,
        description: str = DESCRIPTION,
        timeout: float = 9999.0,
        max_results: int = 5,
        max_batch_size: int = 10,
        batch_interval: float = 0.01,
        cache_file_path: str = "web_search_cache.json", 
    ): 
        self.timeout = timeout
        self.max_results = max_results
        self.max_batch_size = max_batch_size
        self.batch_interval = batch_interval
        self.cache_file_path = cache_file_path

        super().__init__(name=name, description=description)

        self._batcher = self._get_or_create_batcher()

    def _get_or_create_batcher(self) -> _WebBatchRequester:
        # We use a combined key to share batchers: API Key + Cache Path
        # This ensures if different instances use different cache files, they don't mix.
        aggregator_key = f"{self.cache_file_path}"
        
        with self._aggregators_lock:
            batcher = self._aggregators.get(aggregator_key)
            if batcher is None:
                batcher = _WebBatchRequester(
                    timeout=self.timeout,
                    max_batch_size=self.max_batch_size,
                    batch_interval=self.batch_interval,
                    cache_file_path=self.cache_file_path, # Pass path
                )
                self._aggregators[aggregator_key] = batcher
        return batcher

    @property
    def json(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to retrieve relevant information from the web",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": f"Number of results to process (default: {self.max_results})",
                            "minimum": 1,
                            "maximum": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "No relevant information found on the web."

        formatted_results: List[str] = []
        for i, result in enumerate(results, start=1):
            content = result.get("content", "")
            doc_id = result.get("id", f"web_{i}") 

            formatted_result = f"[Document {i}] \n{content}\n"
            formatted_results.append(formatted_result)

        return "\n".join(formatted_results)

    def forward(self, query: str, top_k: Optional[int] = None) -> ToolOutput:
        top_k = min(top_k or self.max_results, 10)
        
        # This returns a Future. 
        # If cache hit, Future is already done. 
        # If miss, it waits for batch processing.
        future = self._batcher.submit(query=query, top_k=top_k)

        docs = future.result(timeout=self.timeout + 5.0)
        if not docs:
            return ToolOutput(
                name=self.name,
                output="No relevant documents found for the query.",
                metadata={
                    "query": query,
                    "num_results": 0,
                    "retriever_type": "web",
                    "cache_hit": False, # Note: We assume false, strict checking would require changing return type of submit
                },
            )
        docs = docs[:top_k]
        formatted_output = self._format_search_results(docs)
        metadata = {
            "query": query,
            "num_results": len(docs),
            "retriever_type": "web",
        }

        return ToolOutput(
            name=self.name,
            output=formatted_output,
            metadata=metadata,
        )

    def __del__(self):
        pass