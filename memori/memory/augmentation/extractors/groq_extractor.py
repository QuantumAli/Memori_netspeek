
import json
import os
from typing import Any

EXTRACTION_SYSTEM_PROMPT = """You are a strict information extraction engine for agent memory.
Only extract information that is explicitly present in the provided conversation.
Do not guess, infer, or fabricate.
Return ONLY valid JSON with the exact schema requested. No markdown, no commentary."""

EXTRACTION_USER_PROMPT_TEMPLATE = """Extract ONLY essential, long-term valuable information from this conversation.

STRICT RULES - FOLLOW EXACTLY:
1. Extract MAX 2-3 facts per message. Fewer is better.
2. Only extract facts that pass this test: "Would I need this info 6 months from now?"
3. NEVER extract:
   - Greetings, pleasantries, or conversational filler
   - Temporary states ("I'm hungry", "I'm tired today")
   - Obvious/implied info (if someone says "I live in NYC", do NOT also add "lives in a city")
   - Rephrased versions of the same fact
   - Generic preferences without specificity
4. DO extract:
   - Identity info (name, location, profession)
   - Specific preferences with detail (not "likes food" but "prefers spicy Thai food")
   - Important relationships (spouse, employer)
   - Key constraints or requirements
5. OUTPUT RULES:
- Output must follow the JSON schema exactly.
- triples must be grounded in explicit text.
- If there are no items for a list, return an empty list (not null).
- If summary is unknown, return null.
- Do not include secrets, API keys, tokens, passwords.

MERGE related info into ONE fact. Example:
  BAD: ["Name is Joe", "Joe lives in NY", "Joe is from New York"]
  GOOD: ["Joe lives in New York"]

INPUT CONVERSATION:
{conversation_json}

OUTPUT SCHEMA:
{{
  "entity": {{
    "facts": ["string"],
    "triples": [
      {{
        "subject": {{"name": "string", "type": "string"}},
        "predicate": "string",
        "object": {{"name": "string", "type": "string"}}
      }}
    ]
  }},
  "process": {{
    "attributes": ["string"]
  }},
  "conversation": {{
    "summary": "string | null"
  }}
}}

REMEMBER: Less is more. Only extract what truly matters.

INPUT CONVERSATION (JSON):
{conversation_json}

OUTPUT JSON SCHEMA (must match):
{{
  "entity": {{
    "facts": ["string"],
    "triples": [
      {{
        "subject": {{"name": "string", "type": "string"}},
        "predicate": "string",
        "object": {{"name": "string", "type": "string"}}
      }}
    ]
  }},
  "process": {{
    "attributes": ["string"]
  }},
  "conversation": {{
    "summary": "string | null"
  }}
}}

Now produce the JSON."""

# Template with reasoning - used when extraction_reasoning is enabled
EXTRACTION_WITH_REASONING_USER_PROMPT_TEMPLATE = """You will be given a conversation with ordered messages and an optional existing summary.

TASK:
Extract durable memory suitable for long-term recall WITH REASONING for each extraction:
1) entity.facts: atomic, user- or entity-specific facts/preferences stated explicitly.
   - For EACH fact, provide reasoning explaining WHY you extracted it.
2) entity.triples: knowledge-graph triples using:
   - subject: {{name, type}}
   - predicate: short verb phrase
   - object: {{name, type}}
   - For EACH triple, provide reasoning explaining WHY you extracted it.
3) process.attributes: non-personal session attributes (project/task constraints) explicitly stated.
4) conversation.summary: concise summary (1–3 sentences) of the conversation content.

STRICT RULES -- FOLLOW EXACTLY:
1. Only extract facts that pass this test: "Would I need this info 6 months from now?"
2. NEVER extract:
   - Greetings, pleasantries, or conversational filler
   - Temporary states ("I'm hungry", "I'm tired today")
   - Obvious/implied info (if someone says "I live in NYC", do NOT also add "lives in a city")
   - Rephrased versions of the same fact
   - Generic preferences without specificity
3. DO extract:
   - Identity info (name, location, profession)
   - Specific preferences with detail (not "likes food" but "prefers spicy Thai food")
   - Important relationships (spouse, employer)
   - Key constraints or requirements
4. OUTPUT RULES:
- Output must follow the JSON schema exactly.
- triples must be grounded in explicit text.
- If there are no items for a list, return an empty list (not null).
- If summary is unknown, return null.
- Do not include secrets, API keys, tokens, passwords.

MERGE related info into ONE fact. Example:
  BAD: ["Name is Joe", "Joe lives in NY", "Joe is from New York"]
  GOOD: ["Joe lives in New York"]

INPUT CONVERSATION (JSON):
{conversation_json}

OUTPUT JSON SCHEMA (must match):
{{
  "entity": {{
    "facts": [
      {{
        "content": "string",
        "reasoning": "string explaining why this fact was extracted, citing conversation text"
      }}
    ],
    "triples": [
      {{
        "subject": {{"name": "string", "type": "string"}},
        "predicate": "string",
        "object": {{"name": "string", "type": "string"}},
        "reasoning": "string explaining why this triple was extracted"
      }}
    ]
  }},
  "process": {{
    "attributes": ["string"]
  }},
  "conversation": {{
    "summary": "string | null"
  }}
}}

Now produce the JSON."""

JSON_REPAIR_PROMPT = """The previous response was not valid JSON. Please fix it and return ONLY valid JSON matching the schema. No markdown, no explanation, just the JSON object."""


class GroqExtractor:
    """Memory extractor using Groq's API.

    This extractor uses Groq's fast inference API to extract memories
    from conversations. It requires the `groq` package to be installed.

    Example usage:
        from memori.memory.augmentation.extractors import GroqExtractor
        from memori import Memori

        # Create extractor with default model
        extractor = GroqExtractor()

        # Or specify a custom model
        extractor = GroqExtractor(model="llama-3.3-70b-versatile")

        # Configure Memori to use the extractor
        memori = Memori(conn)
        memori.config.augmentation_extractor = extractor

        # Enable reasoning (logs why each fact was extracted)
        memori.config.extraction_reasoning = True

    Environment variables:
        GROQ_API_KEY: Your Groq API key (required)
    """

    def __init__(
            self,
            model: str = "llama-3.3-70b-versatile",
            api_key: str | None = None,
            api_key_env: str = "GROQ_API_KEY",
            max_retries: int = 1,
            include_reasoning: bool = False,
    ):
        """Initialize the Groq extractor.

        Args:
            model: The Groq model to use for extraction.
                   Default is "llama-3.3-70b-versatile".
            api_key: Groq API key. If not provided, will read from environment.
            api_key_env: Environment variable name for API key.
            max_retries: Number of retries for JSON repair on parse failure.
            include_reasoning: If True, include reasoning for each extraction.
        """
        self.model = model
        self.api_key = api_key
        self.api_key_env = api_key_env
        self.max_retries = max_retries
        self.include_reasoning = include_reasoning
        self._client = None

    def _get_client(self):
        """Get or create the Groq client."""
        if self._client is not None:
            return self._client

        try:
            from groq import AsyncGroq
        except ImportError:
            raise RuntimeError(
                "The 'groq' package is required for GroqExtractor. "
                "Install it with: pip install groq"
            )

        api_key = self.api_key or os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Groq API key not found. Set the {self.api_key_env} environment "
                "variable or pass api_key to GroqExtractor."
            )

        self._client = AsyncGroq(api_key=api_key)
        return self._client

    async def extract(self, payload: dict, include_reasoning: bool | None = None) -> dict:
        """Extract memories from a conversation payload.

        Args:
            payload: Dictionary containing conversation data with structure:
                {
                    "conversation": {
                        "messages": [{"role": str, "content": str}, ...],
                        "summary": str | None
                    }
                }
            include_reasoning: Override instance setting for including reasoning.

        Returns:
            Extracted memories matching the augmentation response schema.
            When reasoning is enabled, facts include reasoning field.

        Raises:
            RuntimeError: If extraction fails after retries.
        """
        client = self._get_client()

        # Determine if we should include reasoning
        use_reasoning = include_reasoning if include_reasoning is not None else self.include_reasoning

        # Build the conversation JSON for the prompt
        conversation_data = payload.get("conversation", {})
        conversation_json = json.dumps(conversation_data, indent=2)

        # Select the appropriate template
        if use_reasoning:
            user_prompt = EXTRACTION_WITH_REASONING_USER_PROMPT_TEMPLATE.format(
                conversation_json=conversation_json
            )
        else:
            user_prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(
                conversation_json=conversation_json
            )

        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        # First attempt
        response = await self._call_model(client, messages)
        result = self._try_parse_json(response)

        if result is not None:
            return self._normalize_response(result, use_reasoning)

        # Retry with JSON repair instruction
        for _ in range(self.max_retries):
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": JSON_REPAIR_PROMPT})

            response = await self._call_model(client, messages)
            result = self._try_parse_json(response)

            if result is not None:
                return self._normalize_response(result, use_reasoning)

        raise RuntimeError(
            f"Failed to extract valid JSON from model response after "
            f"{self.max_retries + 1} attempts. Last response: {response[:500]}"
        )

    async def _call_model(self, client, messages: list[dict]) -> str:
        """Call the Groq model and return the response content."""
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            # max_tokens=2048, # no reason to have a max limit on this. Will only deteriment the model
        )
        return response.choices[0].message.content or ""

    def _try_parse_json(self, text: str) -> dict | None:
        """Try to parse JSON from text, handling common issues."""
        text = text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _normalize_response(self, result: dict, with_reasoning: bool = False) -> dict:
        """Normalize the response to match expected schema.
        
        Args:
            result: Raw extraction result from LLM.
            with_reasoning: Whether reasoning is included in the response.
            
        Returns:
            Normalized response. When with_reasoning=True, facts and triples
            include reasoning fields.
        """
        entity = result.get("entity", {})
        process = result.get("process", {})
        conversation = result.get("conversation", {})

        raw_facts = entity.get("facts", []) or []
        raw_triples = entity.get("triples", []) or []

        if with_reasoning:
            # Facts are objects with content and reasoning
            facts = []
            facts_reasoning = []
            for fact in raw_facts:
                if isinstance(fact, dict):
                    facts.append(fact.get("content", ""))
                    facts_reasoning.append(fact.get("reasoning", ""))
                else:
                    # Fallback if LLM returned plain strings
                    facts.append(fact)
                    facts_reasoning.append("")

            # Triples include reasoning field
            triples = []
            triples_reasoning = []
            for triple in raw_triples:
                if isinstance(triple, dict):
                    triples.append({
                        "subject": triple.get("subject", {}),
                        "predicate": triple.get("predicate", ""),
                        "object": triple.get("object", {}),
                    })
                    triples_reasoning.append(triple.get("reasoning", ""))

            return {
                "entity": {
                    "facts": facts,
                    "facts_reasoning": facts_reasoning,
                    "triples": triples,
                    "triples_reasoning": triples_reasoning,
                },
                "process": {
                    "attributes": process.get("attributes", []) or [],
                },
                "conversation": {
                    "summary": conversation.get("summary"),
                },
            }
        else:
            # Standard format without reasoning
            return {
                "entity": {
                    "facts": raw_facts,
                    "triples": raw_triples,
                },
                "process": {
                    "attributes": process.get("attributes", []) or [],
                },
                "conversation": {
                    "summary": conversation.get("summary"),
                },
            }

