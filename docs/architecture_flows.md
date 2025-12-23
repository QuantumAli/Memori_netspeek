# Memori Architecture Flows  
  
> A practical guide to how Memori actually works under the hood.  
  
---  
  
## 1. Embedding Space  
  
**Model:** `all-mpnet-base-v2` (768-dimensional vectors)  
  
**Execution:** 100% local. Uses the `sentence-transformers` library—no API calls for embeddings.  
  
```python  
# memori/llm/_embeddings.py  
from sentence_transformers import SentenceTransformer  
encoder = SentenceTransformer("all-mpnet-base-v2")  
embeddings = encoder.encode(texts, convert_to_numpy=True)  
```  
  
The model is cached after first load. Subsequent calls reuse the same instance. This keeps recall fast and free.  
  
---  
  
## 2. API Calls Per Operation  
  
| Operation | API Calls | Why |  
|-----------|-----------|-----|  
| **Recall** (search facts) | 0 | Embeddings are local. FAISS handles similarity search. |  
| **Extraction** (save facts) | 1 | Groq LLM extracts facts from conversation. |  
| **Deduplication** (if enabled) | 1 | Groq LLM decides INSERT/UPDATE/SKIP. |  
| **LLM chat** | 1 | Your actual model response. |  
  
**Typical flow with deduplication enabled:** 3 total API calls    
**Without deduplication:** 2 total API calls  
  
The extraction and deduplication calls both go to Groq's API using your configured model (default: `llama-3.3-70b-versatile`).  
  
---  
  
## 3. Main Features  
  
| Feature              | What It Does                                                      |
| -------------------- | ----------------------------------------------------------------- |
| **Recall**           | Semantic search over stored facts using cosine similarity         |
| **Augmentation**     | Automatic fact extraction from LLM conversations                  |
| **Deduplication**    | LLM-driven decision to skip duplicates or update contradictions   |
| **Multi-DB Support** | SQLite, PostgreSQL, CockroachDB, MySQL, Oracle, MongoDB           |
| **LLM Wrapping**     | Transparently intercepts `client.chat.completions.create()` calls |
| **Session Tracking** | Groups conversations by session with configurable timeout         |
| **Reasoning Logs**   | Optional storage of why each fact was extracted                   |
  
---  
  
## 4. Database Writes  
  
### What Gets Written  
  
| Table                         | Contents                               |
| ----------------------------- | -------------------------------------- |
| `memori_entity`               | Users/entities (by entity_id)          |
| `memori_entity_fact`          | Extracted facts + embeddings           |
| `memori_conversation`         | Conversation metadata + summary        |
| `memori_conversation_message` | Individual messages                    |
| `memori_knowledge_graph`      | Subject-predicate-object triples       |
| `memori_process_attribute`    | Process-level attributes               |
| `memori_extraction_reasoning` | Why each fact was extracted (optional) |
  
### Key Files  
  
| File                                           | Role                                  |
| ---------------------------------------------- | ------------------------------------- |
| `memori/storage/drivers/sqlite/_driver.py`     | SQLite-specific CRUD operations       |
| `memori/storage/drivers/postgresql/_driver.py` | PostgreSQL/CockroachDB operations     |
| `memori/storage/migrations/_sqlite.py`         | SQLite schema definitions             |
| `memori/memory/augmentation/_db_writer.py`     | Async batch writer that queues writes |
  
The driver classes (e.g., `EntityFact.create`) determine how facts are written. Embeddings are stored as binary blobs.  
  
---  
  
## 5. PostgreSQL Integration  
  
Memori auto-detects your database type from the connection object.  
  
**Option A: Environment variable**  
```bash  
export MEMORI_COCKROACHDB_CONNECTION_STRING="postgresql://user:pass@host:5432/db"
```  
  
**Option B: Direct connection**  
```python  
import psycopg  
from memori import Memori  
  
conn = lambda: psycopg.connect("postgresql://user:pass@host:5432/db")  
mem = Memori(conn=conn)  
```  
  
The connection adapter (`memori/storage/adapters/dbapi/_adapter.py`) inspects the connection module name:  
- `psycopg` → PostgreSQL dialect  
- `sqlite` → SQLite dialect  
- `mysql`, `pymysql` → MySQL dialect  
  
The appropriate driver is then loaded from `memori/storage/drivers/`.  
  
---  
  
## 6. Folder Overview  
  
```  
memori/  
├── __init__.py          # Main Memori class, entry point  
├── _config.py           # All configuration options  
├── _search.py           # FAISS-based similarity search  
│  
├── llm/                 # LLM client wrapping  
│   ├── _embeddings.py   # Local embedding generation  
│   ├── _registry.py     # Client registration  
│   └── adapters/        # OpenAI, Anthropic, Google, etc.  
│  
├── memory/              # Memory management  
│   ├── recall.py        # Fact retrieval  
│   ├── _writer.py       # Entity/session/conversation caching  
│   └── augmentation/    # Extraction pipeline  
│       ├── _manager.py      # Async task queue  
│       ├── _deduplication.py # LLM-based dedup logic  
│       ├── _db_writer.py    # Batch database writes  
│       └── extractors/      # GroqExtractor, etc.  
│  
└── storage/             # Database layer  
    ├── _manager.py      # Connection lifecycle    
    ├── _registry.py     # Adapter/driver registry    
    ├── drivers/         # SQLite, PostgreSQL, MySQL, etc.    
    ├── adapters/        # DBAPI, SQLAlchemy, Django adapters    
    └── migrations/      # Schema version control
```  
  
---  
  
## 7. Similarity Score: Functional or Aesthetic?  
  
**Answer: Purely aesthetic (debug output).**  
  
The embedding similarity is computed and printed for visibility:  
  
```  
=== Embedding Similarity Debug ===  
New fact [0]: "Joe lives in NYC"  
  - Similarity 0.8934 with fact #1: "Joe lives in New York" <-- HIGH SIMILARITY
```  
  
But **the LLM makes the final decision**, not the similarity score. The deduplication module calls Groq to decide INSERT/UPDATE/SKIP based on semantic understanding, not a threshold.  
  
The similarity is included so you can:  
- Validate the LLM's decisions  
- Tune your extraction prompts  
- Catch edge cases where high similarity should have triggered a skip  
  
---  
  
## 8. Key Architecture Notes  
  
### For New Contributors  
  
1. **Extraction is async.** After `client.chat.completions.create()` returns, augmentation runs in a background thread. Call `mem.augmentation.wait()` to block until complete.  
  
2. **Facts are embedded at write time.** When a fact is saved, its embedding is computed locally and stored alongside it. Recall never re-embeds.  
  
3. **Deduplication requires existing facts.** On first extraction, there's nothing to compare against—all facts are INSERT. Deduplication only kicks in on subsequent messages.  
  
4. **The driver pattern.** Each database has its own driver (`drivers/sqlite`, `drivers/postgresql`). All implement the same interface. The adapter auto-selects based on your connection.  
  
5. **Extractors are pluggable.** You can swap `GroqExtractor` for any async callable:  
   ```python  
   mem.config.augmentation_extractor = my_custom_extractor  
   ```  
6. **Reasoning is optional.** Enable with `mem.config.extraction_reasoning = True`. This adds ~500 bytes per fact to storage but gives you audit trails.  
  
7. **Session timeout matters.** Conversations group by session. Default timeout is 30 minutes. Same session = same conversation context for deduplication.  
  
---  
  
## Quick Reference: Data Flow  
  
```  
User Message  
     │     
     ▼
┌──────────────────┐  
│  LLM Response    │ ◄─── API Call #1 (your model)  
└──────────────────┘  
     │     
     ▼
┌──────────────────┐  
│  GroqExtractor   │ ◄─── API Call #2 (extraction)  
│  (async thread)  │  
└──────────────────┘  
     │     
     ▼
┌──────────────────┐  
│  Deduplication   │ ◄─── API Call #3 (if enabled)  
│  (LLM decides)   │  
└──────────────────┘  
     │     
     ▼
┌──────────────────┐  
│  Database Write  │ ◄─── Local (SQLite/Postgres)  
│  + Embeddings    │ ◄─── Local (sentence-transformers)  
└──────────────────┘  
```  
