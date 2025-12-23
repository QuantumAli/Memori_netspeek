r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__|_|
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                  [offline fork]
"""

from memori._search import parse_embedding
from memori.llm._embeddings import embed_texts_async
from memori.memory._struct import Memories
from memori.memory.augmentation._base import AugmentationContext, BaseAugmentation
from memori.memory.augmentation._deduplication import deduplicate_facts
from memori.memory.augmentation._registry import Registry


@Registry.register("advanced_augmentation")
class AdvancedAugmentation(BaseAugmentation):
    def __init__(self, config=None, enabled: bool = True):
        super().__init__(config=config, enabled=enabled)

    def _get_conversation_summary(self, driver, conversation_id: str) -> str:
        try:
            conversation = driver.conversation.read(conversation_id)
            if conversation and conversation.get("summary"):
                return conversation["summary"]
        except Exception:
            pass
        return ""

    def _build_extraction_payload(
        self,
        messages: list,
        summary: str,
    ) -> dict:
        """Build payload for local extraction."""
        return {
            "conversation": {
                "messages": messages,
                "summary": summary if summary else None,
            }
        }

    async def _extract_memories(self, payload: dict, include_reasoning: bool = False) -> dict:
        """Extract memories using the configured extractor.

        This method replaces the previous API call to memorilabs.ai.
        It uses a user-provided extractor client to perform local extraction.

        Args:
            payload: The extraction payload containing conversation data.
            include_reasoning: Whether to include reasoning for each extraction.

        Returns:
            A dict matching the augmentation response schema:
            {
                "entity": {"facts": [str], "triples": [...]},
                "process": {"attributes": [str]},
                "conversation": {"summary": str|None}
            }

        Raises:
            RuntimeError: If no extractor is configured.
        """
        if not self.config:
            raise RuntimeError(
                "No config available for augmentation extraction."
            )

        extractor = self.config.augmentation_extractor

        if extractor is None:
            raise RuntimeError(
                "No augmentation extractor configured. "
                "Advanced augmentation requires a local extractor in this offline fork. "
                "Configure one using:\n"
                "  from memori.memory.augmentation.extractors import GroqExtractor\n"
                "  memori.config.augmentation_extractor = GroqExtractor()\n\n"
                "Or provide your own extractor implementing the extract(payload) -> dict method."
            )

        # Support both Protocol-style objects and async callables
        if hasattr(extractor, "extract") and callable(extractor.extract):
            return await extractor.extract(payload, include_reasoning=include_reasoning)
        elif callable(extractor):
            return await extractor(payload)
        else:
            raise RuntimeError(
                "Invalid augmentation extractor. Must be an object with an async "
                "'extract(payload) -> dict' method or an async callable."
            )

    async def process(self, ctx: AugmentationContext, driver) -> AugmentationContext:
        if not ctx.payload.entity_id:
            return ctx
        if not self.config:
            return ctx
        if not ctx.payload.conversation_id:
            return ctx

        summary = self._get_conversation_summary(driver, ctx.payload.conversation_id)

        payload = self._build_extraction_payload(
            ctx.payload.conversation_messages,
            summary,
        )

        # Determine if we should include reasoning
        include_reasoning = getattr(self.config, 'extraction_reasoning', False)

        try:
            api_response = await self._extract_memories(payload, include_reasoning)
        except RuntimeError:
            # Re-raise RuntimeError (no extractor configured)
            raise
        except Exception:
            # Silently fail on other extraction errors (network issues, etc.)
            return ctx

        if not api_response:
            return ctx

        # Get entity ID for deduplication
        entity_id = driver.entity.create(ctx.payload.entity_id)
        if not entity_id:
            return ctx

        # Process response and run deduplication
        memories, reasoning_data = await self._process_with_deduplication(
            api_response, entity_id, ctx.payload.conversation_id, driver, include_reasoning
        )

        if memories is None:
            return ctx

        ctx.data["memories"] = memories
        ctx.data["reasoning"] = reasoning_data

        await self._schedule_entity_writes(ctx, driver, memories, reasoning_data)
        self._schedule_process_writes(ctx, driver, memories)
        self._schedule_conversation_writes(ctx, memories)

        # Store reasoning if enabled
        if include_reasoning and reasoning_data:
            self._schedule_reasoning_writes(ctx, entity_id, reasoning_data)

        return ctx

    async def _process_with_deduplication(
        self,
        api_response: dict,
        entity_id: int,
        conversation_id: int,
        driver,
        include_reasoning: bool,
    ) -> tuple:
        """Process API response with deduplication logic.
        
        Args:
            api_response: Raw extraction response.
            entity_id: Database entity ID.
            conversation_id: Database conversation ID.
            driver: Storage driver.
            include_reasoning: Whether reasoning is included.
            
        Returns:
            Tuple of (Memories, reasoning_data dict).
        """
        entity_data = api_response.get("entity", {})
        facts = entity_data.get("facts", [])
        facts_reasoning = entity_data.get("facts_reasoning", [])
        triples = entity_data.get("triples", [])
        triples_reasoning = entity_data.get("triples_reasoning", [])

        # Generate facts from triples if no explicit facts
        if not facts and triples:
            facts = [
                f"{t['subject']['name']} {t['predicate']} {t['object']['name']}"
                for t in triples
                if t.get("subject") and t.get("predicate") and t.get("object")
            ]
            # Generate reasoning for derived facts if we have triple reasoning
            if include_reasoning and triples_reasoning:
                facts_reasoning = [
                    f"Derived from triple: {tr}" for tr in triples_reasoning
                ]

        if not facts:
            return Memories().configure_from_advanced_augmentation(api_response), {}

        # Generate embeddings for new facts
        fact_embeddings = await embed_texts_async(facts)

        # Get existing facts for this entity
        existing_fact_records = driver.entity_fact.get_embeddings(entity_id, limit=100)
        existing_facts = []
        for record in existing_fact_records:
            fact_id = record["id"]
            raw_embedding = record["content_embedding"]
            
            # Get the fact content
            fact_detail = driver.entity_fact.get_by_id(fact_id)
            if fact_detail:
                try:
                    embedding = parse_embedding(raw_embedding).tolist()
                    existing_facts.append({
                        "id": fact_id,
                        "content": fact_detail["content"],
                        "embedding": embedding,
                    })
                except Exception:
                    continue

        # Run deduplication
        extractor = self.config.augmentation_extractor
        decisions = await deduplicate_facts(
            new_facts=facts,
            new_embeddings=fact_embeddings,
            existing_facts=existing_facts,
            extractor=extractor,
        )

        # Filter facts based on decisions
        facts_to_insert = []
        embeddings_to_insert = []
        updates_to_apply = []  # (fact_id, new_content, new_embedding)
        reasoning_data = {
            "facts": [],
            "decisions": decisions,
            "updates": [],
        }

        for i, decision in enumerate(decisions):
            reasoning_entry = {
                "fact_content": decision.fact_content,
                "reasoning": facts_reasoning[i] if i < len(facts_reasoning) else "",
                "decision": decision.decision,
                "existing_fact_id": decision.existing_fact_id,
                "embedding_similarity": decision.embedding_similarity,
            }
            reasoning_data["facts"].append(reasoning_entry)

            if decision.decision == "INSERT":
                facts_to_insert.append(facts[i])
                embeddings_to_insert.append(fact_embeddings[i])
            elif decision.decision == "UPDATE" and decision.existing_fact_id is not None:
                updates_to_apply.append({
                    "fact_id": decision.existing_fact_id,
                    "new_content": facts[i],
                    "new_embedding": fact_embeddings[i],
                })
            # SKIP: do nothing

        # Store updates for later processing
        reasoning_data["updates"] = updates_to_apply

        # Update the api_response with filtered facts
        api_response["entity"]["facts"] = facts_to_insert
        api_response["entity"]["fact_embeddings"] = embeddings_to_insert

        return Memories().configure_from_advanced_augmentation(api_response), reasoning_data

    async def _process_api_response(self, api_response: dict) -> Memories:
        """Legacy method for backwards compatibility."""
        entity_data = api_response.get("entity", {})
        facts = entity_data.get("facts", [])
        triples = entity_data.get("triples", [])

        if not facts and triples:
            facts = [
                f"{t['subject']['name']} {t['predicate']} {t['object']['name']}"
                for t in triples
                if t.get("subject") and t.get("predicate") and t.get("object")
            ]

        if facts:
            fact_embeddings = await embed_texts_async(facts)
            api_response["entity"]["fact_embeddings"] = fact_embeddings

        return Memories().configure_from_advanced_augmentation(api_response)

    async def _schedule_entity_writes(
        self, ctx: AugmentationContext, driver, memories: Memories, reasoning_data: dict = None
    ):
        if not ctx.payload.entity_id:
            return

        entity_id = driver.entity.create(ctx.payload.entity_id)
        if not entity_id:
            return

        facts_to_write = memories.entity.facts
        embeddings_to_write = memories.entity.fact_embeddings

        if memories.entity.semantic_triples and (
            not facts_to_write or not embeddings_to_write
        ):
            facts_from_triples = [
                f"{triple.subject_name} {triple.predicate} {triple.object_name}"
                for triple in memories.entity.semantic_triples
            ]

            if facts_from_triples:
                embeddings_from_triples = await embed_texts_async(facts_from_triples)
                facts_to_write = (facts_to_write or []) + facts_from_triples
                embeddings_to_write = (
                    embeddings_to_write or []
                ) + embeddings_from_triples

        # Schedule INSERT operations
        if facts_to_write and embeddings_to_write:
            ctx.add_write(
                "entity_fact.create",
                entity_id,
                facts_to_write,
                embeddings_to_write,
            )

        # Schedule UPDATE operations from deduplication
        if reasoning_data and reasoning_data.get("updates"):
            for update in reasoning_data["updates"]:
                ctx.add_write(
                    "entity_fact.update",
                    update["fact_id"],
                    update["new_content"],
                    update["new_embedding"],
                )

        if memories.entity.semantic_triples:
            ctx.add_write(
                "knowledge_graph.create",
                entity_id,
                memories.entity.semantic_triples,
            )

    def _schedule_process_writes(
        self, ctx: AugmentationContext, driver, memories: Memories
    ):
        if not ctx.payload.process_id:
            return

        process_id = driver.process.create(ctx.payload.process_id)
        if process_id and memories.process.attributes:
            ctx.add_write(
                "process_attribute.create", process_id, memories.process.attributes
            )

    def _schedule_conversation_writes(
        self, ctx: AugmentationContext, memories: Memories
    ):
        if not ctx.payload.conversation_id:
            return

        if memories.conversation.summary:
            ctx.add_write(
                "conversation.update",
                ctx.payload.conversation_id,
                memories.conversation.summary,
            )

    def _schedule_reasoning_writes(
        self, ctx: AugmentationContext, entity_id: int, reasoning_data: dict
    ):
        """Schedule writes for extraction reasoning records."""
        if not ctx.payload.conversation_id:
            return

        facts_reasoning = reasoning_data.get("facts", [])
        decisions = reasoning_data.get("decisions", [])

        for i, fact_entry in enumerate(facts_reasoning):
            decision = decisions[i] if i < len(decisions) else None
            
            ctx.add_write(
                "extraction_reasoning.create",
                entity_id,
                ctx.payload.conversation_id,
                fact_entry.get("fact_content", ""),
                fact_entry.get("reasoning", ""),
                "fact",  # extraction_type
                decision.decision if decision else None,
                decision.existing_fact_id if decision else None,
                decision.embedding_similarity if decision else None,
            )
