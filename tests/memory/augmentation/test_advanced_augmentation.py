from unittest.mock import AsyncMock, Mock, patch

import pytest

from memori._config import Config
from memori.memory._struct import Memories
from memori.memory.augmentation._base import AugmentationContext
from memori.memory.augmentation.augmentations.memori._augmentation import (
    AdvancedAugmentation,
)
from memori.memory.augmentation.input import AugmentationInput


class DummyExtractor:
    """Dummy extractor for testing."""

    def __init__(self, response=None):
        self.response = response or {
            "entity": {"facts": [], "triples": []},
            "process": {"attributes": []},
            "conversation": {"summary": None},
        }

    async def extract(self, payload):
        return self.response


@pytest.fixture
def config():
    config = Config()
    config.llm.provider = "openai"
    config.llm.version = "gpt-4"
    config.version = "1.0.0"
    config.storage_config.cockroachdb = False
    config.augmentation_extractor = DummyExtractor()
    return config


@pytest.fixture
def augmentation(config):
    return AdvancedAugmentation(config=config, enabled=True)


@pytest.fixture
def driver():
    driver = Mock()
    driver.conversation.conn.get_dialect.return_value = "postgresql"
    driver.conversation.read.return_value = {"summary": "Previous conversation summary"}
    driver.entity.create.return_value = 1
    driver.process.create.return_value = 1
    return driver


@pytest.fixture
def augmentation_input():
    return AugmentationInput(
        conversation_id="conv-123",
        entity_id="entity-456",
        process_id="process-789",
        conversation_messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ],
        system_prompt="You are a helpful assistant.",
    )


@pytest.mark.asyncio
async def test_process_no_entity_id(augmentation, driver):
    input_payload = AugmentationInput(
        conversation_id="conv-123",
        entity_id=None,
        process_id="process-789",
        conversation_messages=[],
        system_prompt=None,
    )
    ctx = AugmentationContext(payload=input_payload)

    result = await augmentation.process(ctx, driver)

    assert result == ctx
    assert "memories" not in ctx.data


@pytest.mark.asyncio
async def test_process_no_conversation_id(augmentation, driver):
    input_payload = AugmentationInput(
        conversation_id=None,
        entity_id="entity-456",
        process_id="process-789",
        conversation_messages=[],
        system_prompt=None,
    )
    ctx = AugmentationContext(payload=input_payload)

    result = await augmentation.process(ctx, driver)

    assert result == ctx
    assert "memories" not in ctx.data


@pytest.mark.asyncio
async def test_build_extraction_payload(augmentation):
    """Test the extraction payload builder."""
    messages = [{"role": "user", "content": "Test"}]
    summary = "Test summary"

    payload = augmentation._build_extraction_payload(messages, summary)

    assert payload["conversation"]["messages"] == messages
    assert payload["conversation"]["summary"] == summary


@pytest.mark.asyncio
async def test_build_extraction_payload_no_summary(augmentation):
    """Test extraction payload with no summary."""
    messages = [{"role": "user", "content": "Test"}]
    summary = ""

    payload = augmentation._build_extraction_payload(messages, summary)

    assert payload["conversation"]["messages"] == messages
    assert payload["conversation"]["summary"] is None


@pytest.mark.asyncio
async def test_get_conversation_summary_success(augmentation, driver):
    summary = augmentation._get_conversation_summary(driver, "conv-123")

    assert summary == "Previous conversation summary"
    driver.conversation.read.assert_called_once_with("conv-123")


@pytest.mark.asyncio
async def test_get_conversation_summary_no_summary(augmentation, driver):
    driver.conversation.read.return_value = {}

    summary = augmentation._get_conversation_summary(driver, "conv-123")

    assert summary == ""


@pytest.mark.asyncio
async def test_get_conversation_summary_exception(augmentation, driver):
    driver.conversation.read.side_effect = Exception("Database error")

    summary = augmentation._get_conversation_summary(driver, "conv-123")

    assert summary == ""


@pytest.mark.asyncio
async def test_process_api_response_dict_to_memories(augmentation):
    api_response = {
        "entity": {"facts": ["User likes pizza", "User is from NYC"], "triples": []},
        "conversation": {"summary": "Discussed food preferences"},
        "process": {"attributes": ["food", "location"]},
    }

    with patch(
        "memori.memory.augmentation.augmentations.memori._augmentation.embed_texts_async"
    ) as mock_embed:
        mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4]]

        result = await augmentation._process_api_response(api_response)

        assert isinstance(result, Memories)
        assert result.entity.facts == ["User likes pizza", "User is from NYC"]
        assert result.conversation.summary == "Discussed food preferences"


@pytest.mark.asyncio
async def test_process_api_response_triples_to_facts(augmentation):
    api_response = {
        "entity": {
            "facts": [],
            "triples": [
                {
                    "subject": {"name": "User", "type": "Person"},
                    "predicate": "likes",
                    "object": {"name": "Pizza", "type": "Food"},
                },
                {
                    "subject": {"name": "User", "type": "Person"},
                    "predicate": "lives_in",
                    "object": {"name": "NYC", "type": "City"},
                },
            ],
        }
    }

    with patch(
        "memori.memory.augmentation.augmentations.memori._augmentation.embed_texts_async"
    ) as mock_embed:
        mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4]]

        await augmentation._process_api_response(api_response)

        mock_embed.assert_called_once()
        call_args = mock_embed.call_args[0][0]
        assert "User likes Pizza" in call_args
        assert "User lives_in NYC" in call_args


@pytest.mark.asyncio
async def test_schedule_entity_writes(augmentation, driver, augmentation_input):
    ctx = AugmentationContext(payload=augmentation_input)
    memories = Memories()
    memories.entity.facts = ["Fact 1", "Fact 2"]
    memories.entity.fact_embeddings = [[0.1, 0.2], [0.3, 0.4]]

    await augmentation._schedule_entity_writes(ctx, driver, memories)

    assert len(ctx.writes) == 1
    assert ctx.writes[0]["method_path"] == "entity_fact.create"


@pytest.mark.asyncio
async def test_schedule_entity_writes_with_semantic_triples(
    augmentation, driver, augmentation_input
):
    ctx = AugmentationContext(payload=augmentation_input)
    memories = Memories()
    memories.entity.facts = ["Fact 1"]
    memories.entity.fact_embeddings = [[0.1, 0.2]]

    from memori.memory._struct import SemanticTriple

    triple = SemanticTriple()
    triple.subject_name = "User"
    triple.predicate = "likes"
    triple.object_name = "Pizza"
    memories.entity.semantic_triples = [triple]

    await augmentation._schedule_entity_writes(ctx, driver, memories)

    assert len(ctx.writes) == 2
    assert ctx.writes[0]["method_path"] == "entity_fact.create"
    assert ctx.writes[1]["method_path"] == "knowledge_graph.create"


@pytest.mark.asyncio
async def test_schedule_process_writes(augmentation, driver, augmentation_input):
    ctx = AugmentationContext(payload=augmentation_input)
    memories = Memories()
    memories.process.attributes = ["attr1", "attr2"]

    augmentation._schedule_process_writes(ctx, driver, memories)

    assert len(ctx.writes) == 1
    assert ctx.writes[0]["method_path"] == "process_attribute.create"


@pytest.mark.asyncio
async def test_schedule_conversation_writes(augmentation, augmentation_input):
    ctx = AugmentationContext(payload=augmentation_input)
    memories = Memories()
    memories.conversation.summary = "New summary"

    augmentation._schedule_conversation_writes(ctx, memories)

    assert len(ctx.writes) == 1
    assert ctx.writes[0]["method_path"] == "conversation.update"
    assert ctx.writes[0]["args"][0] == "conv-123"


@pytest.mark.asyncio
async def test_schedule_writes_skips_if_no_data(
    augmentation, driver, augmentation_input
):
    ctx = AugmentationContext(payload=augmentation_input)
    memories = Memories()

    await augmentation._schedule_entity_writes(ctx, driver, memories)
    augmentation._schedule_process_writes(ctx, driver, memories)
    augmentation._schedule_conversation_writes(ctx, memories)

    assert len(ctx.writes) == 0


@pytest.mark.asyncio
async def test_extract_memories_with_extractor(config):
    """Test that extract_memories uses the configured extractor."""
    expected_response = {
        "entity": {"facts": ["test fact"], "triples": []},
        "process": {"attributes": []},
        "conversation": {"summary": "test summary"},
    }
    config.augmentation_extractor = DummyExtractor(expected_response)
    augmentation = AdvancedAugmentation(config=config)

    payload = {"conversation": {"messages": [{"role": "user", "content": "test"}]}}
    result = await augmentation._extract_memories(payload)

    assert result == expected_response


@pytest.mark.asyncio
async def test_extract_memories_no_extractor_raises():
    """Test that missing extractor raises RuntimeError."""
    config = Config()
    config.augmentation_extractor = None
    augmentation = AdvancedAugmentation(config=config)

    payload = {"conversation": {"messages": []}}

    with pytest.raises(RuntimeError) as exc_info:
        await augmentation._extract_memories(payload)

    assert "No augmentation extractor configured" in str(exc_info.value)


@pytest.mark.asyncio
async def test_process_with_extractor(config, driver, augmentation_input):
    """Test full process with a configured extractor."""
    response = {
        "entity": {"facts": ["User likes coffee"], "triples": []},
        "process": {"attributes": ["morning routine"]},
        "conversation": {"summary": "Discussing preferences"},
    }
    config.augmentation_extractor = DummyExtractor(response)
    augmentation = AdvancedAugmentation(config=config)

    ctx = AugmentationContext(payload=augmentation_input)

    with patch(
        "memori.memory.augmentation.augmentations.memori._augmentation.embed_texts_async",
        new_callable=AsyncMock,
    ) as mock_embed:
        mock_embed.return_value = [[0.1] * 768]

        result = await augmentation.process(ctx, driver)

        assert "memories" in result.data
        assert result.data["memories"].entity.facts == ["User likes coffee"]
        assert result.data["memories"].conversation.summary == "Discussing preferences"
