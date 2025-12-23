# Memori Offline Fork

This is an offline fork of [MemoriLabs/Memori](https://github.com/MemoriLabs/Memori) that eliminates **all outbound calls to memorilabs.ai** while preserving the core functionality: schema creation, table management, and memory writer pipeline.

## What Was Removed

### Cloud Modules Deleted

| Module | Purpose | Status |
|--------|---------|--------|
| `memori/api/_quota.py` | Cloud quota checking | **Deleted** |
| `memori/api/_sign_up.py` | Cloud account sign-up | **Deleted** |
| `memori/storage/cockroachdb/_cluster_manager.py` | Managed CockroachDB cluster provisioning | **Deleted** |
| `memori/_network.py` | MemoriLabs API client | **Deleted** |
| `memori/memory/_collector.py` | Telemetry collector (POST to /rec) | **Deleted** |

### CLI Commands Stubbed

The following CLI commands now display an error message explaining they are not available:

- `python -m memori quota` → Cloud quota check (removed)
- `python -m memori sign-up <email>` → Cloud sign-up (removed)
- `python -m memori cockroachdb cluster <start|claim|delete>` → Managed cluster (removed)

The `setup` command remains fully functional.

### Error Types Removed

- `QuotaExceededError` → Replaced with `ExtractorNotConfiguredError`
- `MemoriApiError`, `MemoriApiClientError`, `MemoriApiValidationError`, `MemoriApiRequestRejectedError` → Removed (no cloud API)

## Advanced Augmentation: Local Extraction

The original Memori SDK sent conversation data to `api.memorilabs.ai/v1/sdk/augmentation` for memory extraction. This fork replaces that with a **local extractor** that you configure.

### Configuring an Extractor

#### Option 1: Use the built-in Groq Extractor

```python
from memori import Memori
from memori.memory.augmentation.extractors import GroqExtractor

# Set your Groq API key
import os
os.environ["GROQ_API_KEY"] = "your-groq-api-key"

# Create extractor with default model (llama-3.3-70b-versatile)
extractor = GroqExtractor()

# Or specify a different model
extractor = GroqExtractor(model="llama-3.3-70b-versatile")

# Configure Memori
memori = Memori(conn=your_db_connection)
memori.config.augmentation_extractor = extractor
```

#### Option 2: Create a Custom Extractor

Any object with an async `extract(payload) -> dict` method works:

```python
class MyCustomExtractor:
    async def extract(self, payload: dict) -> dict:
        """
        payload contains:
        {
            "conversation": {
                "messages": [{"role": "user/assistant", "content": "..."}],
                "summary": "optional existing summary"
            }
        }
        """
        # Your extraction logic here (call your own LLM, use local models, etc.)
        
        return {
            "entity": {
                "facts": ["fact1", "fact2"],
                "triples": [
                    {
                        "subject": {"name": "User", "type": "person"},
                        "predicate": "prefers",
                        "object": {"name": "dark mode", "type": "preference"}
                    }
                ]
            },
            "process": {
                "attributes": ["project:my-project"]
            },
            "conversation": {
                "summary": "Brief summary of the conversation"
            }
        }

memori.config.augmentation_extractor = MyCustomExtractor()
```

#### Option 3: Use an Async Callable

```python
async def my_extractor(payload: dict) -> dict:
    # Your logic here
    return {"entity": {"facts": [], "triples": []}, ...}

memori.config.augmentation_extractor = my_extractor
```

### What Happens Without an Extractor?

If you don't configure an extractor and the augmentation pipeline runs, you'll get:

```
RuntimeError: No augmentation extractor configured. 
Advanced augmentation requires a local extractor in this offline fork.
Configure one using:
  from memori.memory.augmentation.extractors import GroqExtractor
  memori.config.augmentation_extractor = GroqExtractor()
```

## Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest -q

# Run only the offline fork tests
pytest tests/test_offline_fork.py -v

# Verify no memorilabs.ai references in code
pytest tests/test_offline_fork.py::TestNoMemoriLabsDomains -v
```

## Extraction Prompt

The built-in `GroqExtractor` uses the following prompt for memory extraction:

### System Prompt
```
You are a strict information extraction engine for agent memory.
Only extract information that is explicitly present in the provided conversation.
Do not guess, infer, or fabricate.
Return ONLY valid JSON with the exact schema requested. No markdown, no commentary.
```

### User Prompt Template
```
You will be given a conversation with ordered messages and an optional existing summary.

TASK:
Extract durable memory suitable for long-term recall:
1) entity.facts: atomic, user- or entity-specific facts/preferences stated explicitly.
2) entity.triples: knowledge-graph triples using:
   - subject: {name, type}
   - predicate: short verb phrase
   - object: {name, type}
3) process.attributes: non-personal session attributes (project/task constraints) explicitly stated.
4) conversation.summary: concise summary (1–3 sentences) of the conversation content.

RULES:
- Output must follow the JSON schema exactly.
- facts must be short strings; no duplicates.
- triples must be grounded in explicit text.
- If there are no items for a list, return an empty list (not null).
- If summary is unknown, return null.
- Do not include secrets, API keys, tokens, passwords.

INPUT CONVERSATION (JSON):
<conversation data here>

OUTPUT JSON SCHEMA (must match):
{
  "entity": {
    "facts": ["string"],
    "triples": [
      {
        "subject": {"name": "string", "type": "string"},
        "predicate": "string",
        "object": {"name": "string", "type": "string"}
      }
    ]
  },
  "process": {
    "attributes": ["string"]
  },
  "conversation": {
    "summary": "string | null"
  }
}
```

## Supported Models

The `GroqExtractor` is tested with:
- `llama-3.3-70b-versatile` (default)

You can use any Groq-supported model by passing the `model` parameter.

## Dependencies

This fork adds an optional dependency on the `groq` package for the built-in extractor:

```bash
pip install groq
```

If you're using a custom extractor, you don't need this dependency.

## License

Same as the original Memori project: Apache-2.0

