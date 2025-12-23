r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__|_|
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                  [offline fork]
"""

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from typing import Any, Protocol


class AugmentationExtractor(Protocol):
    """Protocol for augmentation extractors.

    Implement this protocol to provide a custom extractor for augmentation.
    The extractor must have an async `extract` method that takes a payload dict
    and returns the extracted memories.
    """

    async def extract(self, payload: dict) -> dict:
        """Extract memories from a conversation payload.

        Args:
            payload: A dictionary containing:
                - conversation: dict with 'messages' (list) and optional 'summary' (str)

        Returns:
            A dictionary matching the augmentation response schema:
            {
                "entity": {
                    "facts": [str],
                    "triples": [{"subject": {"name": str, "type": str},
                                "predicate": str,
                                "object": {"name": str, "type": str}}]
                },
                "process": {
                    "attributes": [str]
                },
                "conversation": {
                    "summary": str | None
                }
            }
        """
        ...


class Cache:
    def __init__(self):
        self.conversation_id = None
        self.entity_id = None
        self.process_id = None
        self.session_id = None


class Storage:
    def __init__(self):
        self.cockroachdb = False


class Config:
    def __init__(self):
        self.api_key = None
        self.augmentation = None
        self.cache = Cache()
        self.enterprise = False
        self.llm = Llm()
        self.framework = Framework()
        self.platform = Platform()
        self.entity_id = None
        self.process_id = None
        self.raise_final_request_attempt = True
        self.recall_embeddings_limit = 1000
        self.recall_facts_limit = 5
        self.recall_relevance_threshold = 0.1
        self.request_backoff_factor = 1
        self.request_num_backoff = 5
        self.request_secs_timeout = 5
        self.session_id = None
        self.session_timeout_minutes = 30
        self.storage = None
        self.storage_config = Storage()
        self.thread_pool_executor = ThreadPoolExecutor(max_workers=15)
        self.version = version("memori")

        # Augmentation extractor configuration (offline fork)
        # Set this to an object with an async `extract(payload) -> dict` method
        # or an async callable. If None, augmentation will raise RuntimeError.
        self.augmentation_extractor: AugmentationExtractor | Callable[[dict], Any] | None = None

        # Default model for Groq-based extraction (when using GroqExtractor)
        self.augmentation_model: str = "llama-3.3-70b-versatile"

        # Environment variable name for Groq API key
        self.groq_api_key_env: str = "GROQ_API_KEY"

        # Enable extraction reasoning - logs why each fact was extracted
        # When enabled, the extractor will also output reasoning per fact
        self.extraction_reasoning: bool = False

    def is_test_mode(self):
        return os.environ.get("MEMORI_TEST_MODE", None) is not None

    def reset_cache(self):
        self.cache = Cache()
        return self


class Framework:
    def __init__(self):
        self.provider = None


class Platform:
    def __init__(self):
        self.provider = None


class Llm:
    def __init__(self):
        self.provider = None
        self.provider_sdk_version = None
        self.version = None
