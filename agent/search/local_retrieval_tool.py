#!/usr/bin/env python3

import logging
import os
import queue
import threading
import time
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from rllm.tools.tool_base import Tool, ToolOutput

logger = logging.getLogger(__name__)


@dataclass
class _BatchTask:
    query: str
    top_k: int
    future: Future = field(default_factory=Future)


class _BatchRequester:
    """
    Aggregates retrieval queries across tool instances and submits them to the server in batches.
    """

    def __init__(
        self,
        server_url: str,
        timeout: float,
        max_batch_size: int = 256,
        batch_interval: float = 0.02,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.max_batch_size = max_batch_size
        self.batch_interval = batch_interval
        self.queue: queue.Queue[_BatchTask] = queue.Queue()
        self.client = httpx.Client(timeout=timeout)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker,
            name=f"retrieval-batcher-{self.server_url}",
            daemon=True,
        )
        self._worker_thread.start()

    def submit(self, query: str, top_k: int) -> Future:
        task = _BatchTask(query=query, top_k=top_k)
        self.queue.put(task)
        return task.future

    def close(self):
        self._stop_event.set()
        self._worker_thread.join(timeout=1.0)
        try:
            self.client.close()
        except Exception:  # noqa: BLE001
            pass

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
                logger.exception("Batch retrieval failed: %s", exc)
                for task_item in batch:
                    if not task_item.future.done():
                        task_item.future.set_exception(exc)
            finally:
                for _ in batch:
                    self.queue.task_done()

    def _process_batch(self, batch: List[_BatchTask]):
        payload = {
            "queries": [task.query for task in batch],
            "top_k": [task.top_k for task in batch],
        }
        # print("Batch Len: ", len(batch))
        response = self.client.post(f"{self.server_url}/retrieve_batch", json=payload)
        if not response.is_success:
            error_message = f"Retrieval server returned status code {response.status_code}"
            try:
                error_data = response.json()
                detail = error_data.get("detail") or error_data.get("error")
                if detail:
                    error_message += f": {detail}"
            except Exception:  # noqa: BLE001
                if response.text:
                    error_message += f": {response.text}"
            exception = RuntimeError(error_message)
            for task in batch:
                if not task.future.done():
                    task.future.set_exception(exception)
            return

        try:
            data = response.json()
        except ValueError as exc:
            for task in batch:
                if not task.future.done():
                    task.future.set_exception(exc)
            return

        results = data.get("results", [])
        if not isinstance(results, list) or len(results) != len(batch):
            exception = RuntimeError("Retrieval server returned malformed results.")
            for task in batch:
                if not task.future.done():
                    task.future.set_exception(exception)
            return

        for task, docs in zip(batch, results, strict=True):
            if not isinstance(docs, list):
                docs = []
            if not task.future.done():
                task.future.set_result(docs)


class LocalRetrievalTool(Tool):
    """
    A tool for dense search using the local retrieval server that batches queries across threads.
    """

    NAME = "local_search"
    DESCRIPTION = "Search for information using a dense retrieval server with Wikipedia corpus"

    _aggregators: Dict[str, _BatchRequester] = {}
    _aggregators_lock = threading.Lock()

    def __init__(
        self,
        name: str = NAME,
        description: str = DESCRIPTION,
        server_url: Optional[str] = "http://127.0.0.1:9889",
        timeout: float = 60.0,
        max_results: int = 5,
        max_batch_size: int = 256,
        batch_interval: float = 0.02,
    ):
        if server_url is None:
            server_url = os.environ.get("RETRIEVAL_SERVER_URL", "http://127.0.0.1:9889")

        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.max_results = max_results
        self.max_batch_size = max_batch_size
        self.batch_interval = batch_interval

        super().__init__(name=name, description=description)

        self._batcher = self._get_or_create_batcher()
        # self._test_connection()

    def _get_or_create_batcher(self) -> _BatchRequester:
        with self._aggregators_lock:
            batcher = self._aggregators.get(self.server_url)
            if batcher is None:
                batcher = _BatchRequester(
                    server_url=self.server_url,
                    timeout=self.timeout,
                    max_batch_size=self.max_batch_size,
                    batch_interval=self.batch_interval,
                )
                self._aggregators[self.server_url] = batcher
        return batcher

    def _test_connection(self):
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.server_url}/health")
            if response.status_code == 200:
                logger.info("Successfully connected to retrieval server at %s", self.server_url)
            else:
                logger.warning(
                    "Retrieval server at %s returned status code %s",
                    self.server_url,
                    response.status_code,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not connect to retrieval server: %s", exc)

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
                            "description": "Search query to retrieve relevant documents",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": f"Number of results to return (default: {self.max_results})",
                            "minimum": 1,
                            "maximum": 50,
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "No relevant documents found."

        formatted_results: List[str] = []
        for i, result in enumerate(results[: self.max_results], start=1):
            doc_id = result.get("id", f"doc_{i}")
            content = result.get("content") or result.get("contents") or ""
            score = result.get("score", 0.0)

            # if len(content) > 600:
            #     content = content[:600] + "..."

            formatted_result = f"[Document {i}] (ID: {doc_id}, Score: {score:.3f})\n{content}\n"
            formatted_results.append(formatted_result)

        return "\n".join(formatted_results)

    def forward(self, query: str, top_k: Optional[int] = None) -> ToolOutput:
        top_k = min(top_k or self.max_results, 50)
        future = self._batcher.submit(query=query, top_k=top_k)

        try:
            docs = future.result(timeout=self.timeout + self.batch_interval + 1.0)
        except FutureTimeoutError:
            return ToolOutput(
                name=self.name,
                error=f"Batch retrieval timed out after {self.timeout} seconds.",
            )
        except Exception as exc:  # noqa: BLE001
            return ToolOutput(
                name=self.name,
                error=f"Retrieval failed: {exc}",
            )

        if not docs:
            return ToolOutput(
                name=self.name,
                output="No relevant documents found for the query.",
                metadata={
                    "query": query,
                    "num_results": 0,
                    "retriever_type": "dense",
                    "server_url": self.server_url,
                },
            )

        formatted_output = self._format_search_results(docs)
        metadata = {
            "query": query,
            "num_results": len(docs),
            "retriever_type": "dense",
            "server_url": self.server_url,
        }

        return ToolOutput(
            name=self.name,
            output=formatted_output,
            metadata=metadata,
        )

    def __del__(self):
        try:
            if hasattr(self, "_batcher") and self._batcher is not None:
                with self._aggregators_lock:
                    # Shared batcher instances remain alive; do not close here.
                    pass
        except Exception:  # noqa: BLE001
            pass


def create_local_retrieval_tool(
    server_url: str = "http://127.0.0.1:9889",
    max_results: int = 5,
    timeout: float = 60.0,
    max_batch_size: int = 256,
    batch_interval: float = 0.02,
) -> LocalRetrievalTool:
    return LocalRetrievalTool(
        server_url=server_url,
        max_results=max_results,
        timeout=timeout,
        max_batch_size=max_batch_size,
        batch_interval=batch_interval,
    )