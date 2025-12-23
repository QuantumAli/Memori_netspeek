r"""
Deduplication module for memory extraction.
Determines whether extracted facts should be inserted, updated, or skipped.
"""

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class DeduplicationDecision:
    """Result of deduplication decision for a single fact."""
    fact_content: str
    decision: str  # 'INSERT', 'UPDATE', or 'SKIP'
    reasoning: str
    existing_fact_id: int | None = None  # For UPDATE, the fact being replaced
    existing_fact_content: str | None = None  # For UPDATE/SKIP, the similar fact
    embedding_similarity: float | None = None  # Debug: similarity score


DEDUPLICATION_SYSTEM_PROMPT = """You are a memory deduplication engine.
Your task is to compare new facts against existing facts and decide:
- INSERT: New information that should be added
- UPDATE: Information that contradicts/updates an existing fact (specify which)
- SKIP: Duplicate information that already exists

Be precise and conservative. Only mark as SKIP if the information is truly redundant.
Only mark as UPDATE if the new fact directly contradicts or supersedes an existing one.
Return ONLY valid JSON with the exact schema requested. No markdown, no commentary."""

DEDUPLICATION_USER_PROMPT_TEMPLATE = """Compare these new facts against existing facts for this entity.

EXISTING FACTS (already in memory):
{existing_facts_json}

NEW FACTS (just extracted from conversation):
{new_facts_json}

For EACH new fact, decide:
- INSERT: This is genuinely new information not covered by existing facts
- UPDATE: This contradicts/supersedes an existing fact (specify the existing_fact_id)
- SKIP: This is essentially the same as an existing fact (specify the existing_fact_id)

OUTPUT JSON SCHEMA (must match):
{{
  "decisions": [
    {{
      "new_fact_index": 0,
      "decision": "INSERT" | "UPDATE" | "SKIP",
      "reasoning": "Brief explanation of why this decision was made",
      "existing_fact_id": null | <id of the existing fact for UPDATE/SKIP>
    }}
  ]
}}

Now produce the JSON."""


def compute_embedding_similarities(
    new_embeddings: list[list[float]],
    existing_embeddings: list[tuple[int, list[float]]],
) -> list[list[tuple[int, float]]]:
    """Compute cosine similarities between new and existing embeddings.
    
    Args:
        new_embeddings: List of embeddings for new facts.
        existing_embeddings: List of (fact_id, embedding) tuples for existing facts.
        
    Returns:
        For each new embedding, a sorted list of (existing_fact_id, similarity) tuples.
    """
    if not new_embeddings or not existing_embeddings:
        return [[] for _ in new_embeddings]

    # Get expected dimension from new embeddings
    expected_dim = len(new_embeddings[0]) if new_embeddings else 768

    # Filter and stack existing embeddings - ensure consistent dimensions
    existing_ids = []
    existing_list = []
    for fact_id, emb in existing_embeddings:
        if emb is not None and len(emb) == expected_dim:
            existing_ids.append(fact_id)
            existing_list.append(emb)

    if not existing_list:
        return [[] for _ in new_embeddings]

    existing_array = np.array(existing_list, dtype=np.float32)
    
    # Normalize existing embeddings
    existing_norms = np.linalg.norm(existing_array, axis=1, keepdims=True)
    existing_normalized = existing_array / np.maximum(existing_norms, 1e-10)

    results = []
    for new_emb in new_embeddings:
        if len(new_emb) != expected_dim:
            results.append([])
            continue
            
        new_array = np.array([new_emb], dtype=np.float32)
        new_norm = np.linalg.norm(new_array)
        new_normalized = new_array / max(new_norm, 1e-10)
        
        # Cosine similarity
        similarities = np.dot(existing_normalized, new_normalized.T).flatten()
        
        # Create sorted list of (id, similarity)
        id_sim_pairs = list(zip(existing_ids, similarities.tolist()))
        id_sim_pairs.sort(key=lambda x: x[1], reverse=True)
        results.append(id_sim_pairs)
    
    return results


async def deduplicate_facts(
    new_facts: list[str],
    new_embeddings: list[list[float]],
    existing_facts: list[dict],  # [{id, content, embedding}, ...]
    extractor,  # GroqExtractor or similar
    similarity_threshold: float = 0.85,
) -> list[DeduplicationDecision]:
    """Deduplicate new facts against existing facts using LLM.
    
    Args:
        new_facts: List of new fact strings to deduplicate.
        new_embeddings: Embeddings for the new facts.
        existing_facts: List of existing fact dicts with id, content, embedding.
        extractor: Extractor instance to use for LLM calls.
        similarity_threshold: Threshold for printing similarity debug info.
        
    Returns:
        List of DeduplicationDecision for each new fact.
    """
    if not new_facts:
        return []

    if not existing_facts:
        # No existing facts, all are INSERT
        return [
            DeduplicationDecision(
                fact_content=fact,
                decision="INSERT",
                reasoning="No existing facts to compare against",
                embedding_similarity=None,
            )
            for fact in new_facts
        ]

    # Compute embedding similarities for debug output
    existing_embeddings = []
    for ef in existing_facts:
        emb = ef.get("embedding")
        if emb is not None:
            existing_embeddings.append((ef["id"], emb))

    similarities = compute_embedding_similarities(new_embeddings, existing_embeddings)
    
    # Print debug info about embedding similarities
    print("\n=== Embedding Similarity Debug ===")
    for i, (fact, sims) in enumerate(zip(new_facts, similarities)):
        print(f"\nNew fact [{i}]: \"{fact}\"")
        if sims:
            top_3 = sims[:3]
            for existing_id, sim in top_3:
                existing_content = next(
                    (ef["content"] for ef in existing_facts if ef["id"] == existing_id),
                    "?"
                )
                marker = " <-- HIGH SIMILARITY" if sim >= similarity_threshold else ""
                print(f"  - Similarity {sim:.4f} with fact #{existing_id}: \"{existing_content}\"{marker}")
        else:
            print("  - No existing embeddings to compare")
    print("=== End Similarity Debug ===\n")

    # Build prompts for LLM deduplication
    existing_facts_json = json.dumps(
        [{"id": ef["id"], "content": ef["content"]} for ef in existing_facts],
        indent=2
    )
    new_facts_json = json.dumps(
        [{"index": i, "content": fact} for i, fact in enumerate(new_facts)],
        indent=2
    )
    
    user_prompt = DEDUPLICATION_USER_PROMPT_TEMPLATE.format(
        existing_facts_json=existing_facts_json,
        new_facts_json=new_facts_json,
    )
    
    messages = [
        {"role": "system", "content": DEDUPLICATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    
    # Call LLM for deduplication decision
    client = extractor._get_client()
    response = await extractor._call_model(client, messages)
    result = extractor._try_parse_json(response)
    
    if result is None:
        # Fallback: treat all as INSERT if parsing fails
        print(f"WARNING: Failed to parse deduplication response, treating all as INSERT")
        return [
            DeduplicationDecision(
                fact_content=fact,
                decision="INSERT",
                reasoning="Deduplication parsing failed, defaulting to INSERT",
                embedding_similarity=similarities[i][0][1] if similarities[i] else None,
            )
            for i, fact in enumerate(new_facts)
        ]

    # Parse decisions
    decisions = []
    raw_decisions = result.get("decisions", [])
    
    for i, fact in enumerate(new_facts):
        # Find the decision for this fact
        fact_decision = next(
            (d for d in raw_decisions if d.get("new_fact_index") == i),
            None
        )
        
        if fact_decision is None:
            # No decision found, default to INSERT
            decisions.append(DeduplicationDecision(
                fact_content=fact,
                decision="INSERT",
                reasoning="No decision provided by LLM, defaulting to INSERT",
                embedding_similarity=similarities[i][0][1] if similarities[i] else None,
            ))
        else:
            decision = fact_decision.get("decision", "INSERT").upper()
            existing_id = fact_decision.get("existing_fact_id")
            existing_content = None
            
            if existing_id is not None:
                existing_content = next(
                    (ef["content"] for ef in existing_facts if ef["id"] == existing_id),
                    None
                )
            
            # Get top similarity for this fact
            top_similarity = similarities[i][0][1] if similarities[i] else None
            
            decisions.append(DeduplicationDecision(
                fact_content=fact,
                decision=decision,
                reasoning=fact_decision.get("reasoning", ""),
                existing_fact_id=existing_id,
                existing_fact_content=existing_content,
                embedding_similarity=top_similarity,
            ))
    
    # Print decision summary
    print("\n=== Deduplication Decisions ===")
    for d in decisions:
        if d.decision == "INSERT":
            print(f"  [INSERT] \"{d.fact_content}\" - {d.reasoning}")
        elif d.decision == "UPDATE":
            print(f"  [UPDATE] \"{d.fact_content}\" replaces #{d.existing_fact_id} \"{d.existing_fact_content}\" - {d.reasoning}")
        else:  # SKIP
            print(f"  [SKIP] \"{d.fact_content}\" duplicates #{d.existing_fact_id} \"{d.existing_fact_content}\" - {d.reasoning}")
    print("=== End Decisions ===\n")
    
    return decisions

