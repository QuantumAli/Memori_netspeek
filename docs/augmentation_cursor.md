# Memori Augmentation Flow

This document describes the exact execution path from when you call `client.chat.completions.create()` to when memories are written to the database.

---

## Overview

```
User Query → Wrapped Client → Post-Response Handler → Augmentation Manager → GroqExtractor → DB Writer → SQLite
```

---

## Phase 1: Client Wrapping (Setup)

**When**: `mem.llm.register(client)` is called  
**File**: `memori/llm/_clients.py` → `OpenAi.register()`

The OpenAI client's `chat.completions.create` method is replaced with a wrapped version:

```
client.chat.completions.create → Invoke(config, original_method).invoke
```

**Key code path**:
1. `memori/llm/_registry.py` → `register_llm()` identifies the client type
2. `memori/llm/_clients.py` → `OpenAi.register()` wraps the method
3. `memori/llm/_base.py` → `BaseClient._wrap_method()` creates the `Invoke` wrapper

---

## Phase 2: Query Execution

**When**: `client.chat.completions.create(messages=[...])` is called  
**File**: `memori/llm/_invoke.py` → `Invoke.invoke()`

### Step 2.1: Pre-processing
```python
kwargs = self.inject_conversation_messages(
    self.inject_recalled_facts(self.configure_for_streaming_usage(kwargs))
)
```
- Injects recalled facts into system prompt (if any exist)
- Configures streaming options

### Step 2.2: Execute Original Method
```python
raw_response = self._method(**kwargs)
```
- Calls the original `client.chat.completions.create`
- Waits for LLM response

### Step 2.3: Post-Response Handler
```python
self.handle_post_response(kwargs, start, raw_response)
```

---

## Phase 3: Post-Response Processing

**File**: `memori/llm/_base.py` → `BaseInvoke.handle_post_response()`

### Step 3.1: Format Payload (lines 619-629)
Extracts conversation data including:
- Entity ID and Process ID
- Query messages
- Response content

### Step 3.2: Create Augmentation Input (lines 675-681)
```python
augmentation_input = AugmentationInput(
    conversation_id=self.config.cache.conversation_id,
    entity_id=self.config.entity_id,
    process_id=self.config.process_id,
    conversation_messages=messages_for_aug,
    system_prompt=system_prompt,
)
```

### Step 3.3: Enqueue for Async Processing (line 682)
```python
self.config.augmentation.enqueue(augmentation_input)
```

---

## Phase 4: Augmentation Manager

**File**: `memori/memory/augmentation/_manager.py` → `Manager.enqueue()`

### Step 4.1: Submit to Async Runtime (lines 84-88)
```python
future = asyncio.run_coroutine_threadsafe(
    self._process_augmentations(input_data), runtime.loop
)
```
- Runs in a separate daemon thread (`memori-augmentation`)
- Non-blocking - returns immediately to caller

### Step 4.2: Process Augmentations (lines 100-126)
```python
async def _process_augmentations(self, input_data):
    ctx = AugmentationContext(payload=input_data)
    with connection_context(self.conn_factory) as (conn, adapter, driver):
        for aug in self.augmentations:
            ctx = await aug.process(ctx, driver)
        if ctx.writes:
            self._enqueue_writes(ctx.writes)
```

---

## Phase 5: Advanced Augmentation

**File**: `memori/memory/augmentation/augmentations/memori/_augmentation.py` → `AdvancedAugmentation.process()`

### Step 5.1: Build Extraction Payload (lines 119-122)
```python
payload = self._build_extraction_payload(
    ctx.payload.conversation_messages,
    summary,
)
```

Creates:
```json
{
  "conversation": {
    "messages": [{"role": "user", "content": "..."}, ...],
    "summary": "..." 
  }
}
```

### Step 5.2: Call Extractor (line 125)
```python
api_response = await self._extract_memories(payload)
```

This calls `config.augmentation_extractor.extract(payload)` → **GroqExtractor**

### Step 5.3: Process Response & Generate Embeddings (lines 146-162)
```python
async def _process_api_response(self, api_response):
    facts = entity_data.get("facts", [])
    if facts:
        fact_embeddings = await embed_texts_async(facts)
    return Memories().configure_from_advanced_augmentation(api_response)
```

### Step 5.4: Schedule Database Writes (lines 140-142)
```python
await self._schedule_entity_writes(ctx, driver, memories)
self._schedule_process_writes(ctx, driver, memories)
self._schedule_conversation_writes(ctx, memories)
```

**Writes scheduled**:
| Method Path | Data Written |
|-------------|--------------|
| `entity_fact.create` | Facts + embeddings |
| `knowledge_graph.create` | Semantic triples |
| `process_attribute.create` | Process attributes |
| `conversation.update` | Conversation summary |

---

## Phase 6: GroqExtractor

**File**: `memori/memory/augmentation/extractors/groq_extractor.py` → `GroqExtractor.extract()`

### Step 6.1: Build Prompt (lines 160-171)
Uses two prompts:
- **System**: Strict JSON extraction instructions
- **User**: Conversation data + expected JSON schema

### Step 6.2: Call Groq API (lines 174, 196-204)
```python
response = await client.chat.completions.create(
    model=self.model,  # "llama-3.3-70b-versatile"
    messages=messages,
    temperature=0.1,
    max_tokens=2048,
)
```

### Step 6.3: Parse & Normalize Response (lines 175-178, 225-242)
```python
result = self._try_parse_json(response)
return self._normalize_response(result)
```

**Output schema**:
```json
{
  "entity": {
    "facts": ["User's name is Joe", "User lives in New York"],
    "triples": [
      {
        "subject": {"name": "Joe", "type": "person"},
        "predicate": "lives in",
        "object": {"name": "New York", "type": "location"}
      }
    ]
  },
  "process": {
    "attributes": ["preference:pizza"]
  },
  "conversation": {
    "summary": "User introduced themselves and shared food preferences"
  }
}
```

---

## Phase 7: Database Writing

**File**: `memori/memory/augmentation/_db_writer.py` → `DbWriterRuntime`

### Step 7.1: Writes Enqueued (Manager lines 128-137)
```python
for write_op in writes:
    task = WriteTask(method_path=write_op["method_path"], args=write_op["args"])
    db_writer.enqueue_write(task)
```

### Step 7.2: Batch Processing (lines 84-118)
- Runs in separate daemon thread (`memori-db-writer`)
- Collects writes into batches (max 100, timeout 0.1s)
- Executes batch within a database transaction

### Step 7.3: Task Execution (lines 27-41)
```python
def execute(self, driver):
    method = self._resolve_method(driver, self.method_path)
    # e.g., driver.entity_fact.create(entity_id, facts, embeddings)
    return method(*self.args, **self.kwargs)
```

### Step 7.4: SQLite Driver Methods Called

**File**: `memori/storage/drivers/sqlite/_driver.py`

| Method | Table | Action |
|--------|-------|--------|
| `Entity.create()` | `memori_entity` | Insert entity record |
| `EntityFact.create()` | `memori_entity_fact` | Insert facts with embeddings |
| `KnowledgeGraph.create()` | `memori_knowledge_graph` | Insert semantic triples |
| `Process.create()` | `memori_process` | Insert process record |
| `ProcessAttribute.create()` | `memori_process_attribute` | Insert process attributes |
| `Conversation.update()` | `memori_conversation` | Update summary |

---

## Thread Model

```
Main Thread                    Augmentation Thread           DB Writer Thread
     │                              │                              │
     │ client.chat.create()         │                              │
     │──────────────────────────────│                              │
     │                              │                              │
     │ enqueue(input)               │                              │
     │─────────────────────────────>│                              │
     │                              │ await aug.process()          │
     │ return response              │                              │
     │<─────────────────────────────│ await extractor.extract()    │
     │                              │─────────────────────────────>│ (Groq API)
     │                              │<─────────────────────────────│
     │                              │                              │
     │                              │ enqueue_writes()             │
     │                              │─────────────────────────────>│
     │                              │                              │ batch & execute
     │                              │                              │ driver.entity_fact.create()
     │                              │                              │ commit()
```

---

## File Reference

| File | Purpose |
|------|---------|
| `memori/llm/_invoke.py` | Wraps LLM calls, triggers post-response |
| `memori/llm/_base.py` | `handle_post_response()` creates & enqueues augmentation |
| `memori/memory/augmentation/_manager.py` | Manages async processing, coordinates writes |
| `memori/memory/augmentation/augmentations/memori/_augmentation.py` | Extracts memories, schedules DB writes |
| `memori/memory/augmentation/extractors/groq_extractor.py` | Calls Groq API for memory extraction |
| `memori/memory/augmentation/_db_writer.py` | Batches and executes database writes |
| `memori/storage/drivers/sqlite/_driver.py` | SQLite-specific SQL operations |

