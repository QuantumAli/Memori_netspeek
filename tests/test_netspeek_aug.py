"""
Tests for the offline fork of Memori.

These tests verify that:
1. No memorilabs.ai domains appear in runtime code
2. No HTTP calls to memorilabs.ai are made during runtime
3. The local extraction works correctly
4. CLI cloud commands are properly removed/stubbed
"""

import ast
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestNoMemoriLabsDomains:
    """Test A: Verify no MemoriLabs domains in runtime module code."""

    FORBIDDEN_STRINGS = [
        "api.memorilabs.ai",
        "staging-api.memorilabs.ai",
        "memorilabs.ai/rec",
    ]

    # Allowed contexts where memorilabs.ai can appear (docstrings/comments explaining removal)
    ALLOWED_CONTEXTS = [
        "This method replaces the previous API call to memorilabs.ai",
        "Cloud features (quota, sign-up, cockroachdb cluster management)",
        "no connections to memorilabs.ai",
    ]

    def get_memori_python_files(self):
        """Get all Python files in the memori package."""
        memori_dir = Path(__file__).parent.parent / "memori"
        return list(memori_dir.rglob("*.py"))

    def test_no_api_endpoints_in_code(self):
        """Assert no string literals contain memorilabs.ai API endpoints."""
        violations = []

        for py_file in self.get_memori_python_files():
            content = py_file.read_text(encoding="utf-8")

            for forbidden in self.FORBIDDEN_STRINGS:
                if forbidden in content:
                    # Check if it's in an allowed context
                    lines_with_forbidden = [
                        line
                        for line in content.split("\n")
                        if forbidden in line
                    ]

                    for line in lines_with_forbidden:
                        # Skip if in allowed context (docstrings explaining the change)
                        if not any(ctx in line for ctx in self.ALLOWED_CONTEXTS):
                            violations.append(
                                f"{py_file.relative_to(py_file.parent.parent)}: "
                                f"contains '{forbidden}' in: {line.strip()[:100]}"
                            )

        assert not violations, (
            f"Found {len(violations)} violations with memorilabs.ai domains:\n"
            + "\n".join(violations)
        )

    def test_no_network_module(self):
        """Verify the network module has been removed."""
        network_path = Path(__file__).parent.parent / "memori" / "_network.py"
        assert not network_path.exists(), "_network.py should be deleted"

    def test_no_collector_module(self):
        """Verify the collector module has been removed."""
        collector_path = (
            Path(__file__).parent.parent / "memori" / "memory" / "_collector.py"
        )
        assert not collector_path.exists(), "_collector.py should be deleted"

    def test_no_quota_module(self):
        """Verify the quota module has been removed."""
        quota_path = Path(__file__).parent.parent / "memori" / "api" / "_quota.py"
        assert not quota_path.exists(), "api/_quota.py should be deleted"

    def test_no_signup_module(self):
        """Verify the sign-up module has been removed."""
        signup_path = Path(__file__).parent.parent / "memori" / "api" / "_sign_up.py"
        assert not signup_path.exists(), "api/_sign_up.py should be deleted"

    def test_no_cluster_manager_module(self):
        """Verify the cluster manager module has been removed."""
        cluster_path = (
            Path(__file__).parent.parent
            / "memori"
            / "storage"
            / "cockroachdb"
            / "_cluster_manager.py"
        )
        assert not cluster_path.exists(), (
            "storage/cockroachdb/_cluster_manager.py should be deleted"
        )


class TestRuntimeNetworkGuard:
    """Test B: Runtime network guard to ensure no memorilabs.ai requests."""

    def _create_network_guard(self):
        """Create a guard that raises if URL contains memorilabs.ai."""

        def guard(self, method, url, *args, **kwargs):
            if "memorilabs.ai" in str(url):
                raise AssertionError(
                    f"Attempted to make request to memorilabs.ai: {method} {url}"
                )
            # Return a mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {}
            return mock_response

        return guard

    @pytest.mark.asyncio
    async def test_config_init_no_network_calls(self):
        """Test that Config initialization makes no network calls."""
        with patch("requests.Session.request", self._create_network_guard()):
            from memori._config import Config

            config = Config()
            assert config is not None

    @pytest.mark.asyncio
    async def test_augmentation_with_dummy_extractor_no_network_calls(self):
        """Test that augmentation with a dummy extractor makes no memorilabs.ai calls."""

        # Create a dummy extractor
        class DummyExtractor:
            async def extract(self, payload):
                return {
                    "entity": {
                        "facts": ["test fact"],
                        "triples": [],
                    },
                    "process": {
                        "attributes": [],
                    },
                    "conversation": {
                        "summary": "Test summary",
                    },
                }

        with patch("requests.Session.request", self._create_network_guard()):
            from memori._config import Config
            from memori.memory.augmentation._base import AugmentationContext
            from memori.memory.augmentation.augmentations.memori._augmentation import (
                AdvancedAugmentation,
            )
            from memori.memory.augmentation.input import AugmentationInput

            config = Config()
            config.augmentation_extractor = DummyExtractor()

            augmentation = AdvancedAugmentation(config=config)

            payload = AugmentationInput(
                entity_id="user123",
                process_id="test-process",
                conversation_id="1",
                conversation_messages=[{"role": "user", "content": "test"}],
                system_prompt=None,
            )
            ctx = AugmentationContext(payload=payload)

            mock_driver = MagicMock()
            mock_driver.conversation.conn.get_dialect.return_value = "postgresql"
            mock_driver.conversation.read.return_value = None
            mock_driver.entity.create.return_value = 1
            mock_driver.process.create.return_value = 1

            # Mock embed_texts_async to avoid actual embedding calls
            with patch(
                "memori.memory.augmentation.augmentations.memori._augmentation.embed_texts_async",
                new_callable=AsyncMock,
            ) as mock_embed:
                mock_embed.return_value = [[0.1] * 768]
                result = await augmentation.process(ctx, mock_driver)
                assert result is not None


class TestLocalExtractionContract:
    """Test C: Advanced Augmentation local extraction contract."""

    @pytest.mark.asyncio
    async def test_extraction_with_dummy_extractor(self):
        """Test that extraction works with a dummy extractor."""
        expected_response = {
            "entity": {
                "facts": ["User prefers dark mode", "User lives in NYC"],
                "triples": [
                    {
                        "subject": {"name": "User", "type": "person"},
                        "predicate": "lives in",
                        "object": {"name": "NYC", "type": "location"},
                    }
                ],
            },
            "process": {
                "attributes": ["project:memori-fork"],
            },
            "conversation": {
                "summary": "Discussion about user preferences",
            },
        }

        class DummyExtractor:
            async def extract(self, payload):
                return expected_response

        from memori._config import Config
        from memori.memory.augmentation._base import AugmentationContext
        from memori.memory.augmentation.augmentations.memori._augmentation import (
            AdvancedAugmentation,
        )
        from memori.memory.augmentation.input import AugmentationInput

        config = Config()
        config.augmentation_extractor = DummyExtractor()

        augmentation = AdvancedAugmentation(config=config)

        payload = AugmentationInput(
            entity_id="user123",
            process_id="test-process",
            conversation_id="conv-1",
            conversation_messages=[
                {"role": "user", "content": "I prefer dark mode and live in NYC"},
                {"role": "assistant", "content": "Got it!"},
            ],
            system_prompt=None,
        )
        ctx = AugmentationContext(payload=payload)

        mock_driver = MagicMock()
        mock_driver.conversation.conn.get_dialect.return_value = "postgresql"
        mock_driver.conversation.read.return_value = None
        mock_driver.entity.create.return_value = 1
        mock_driver.process.create.return_value = 1

        with patch(
            "memori.memory.augmentation.augmentations.memori._augmentation.embed_texts_async",
            new_callable=AsyncMock,
        ) as mock_embed:
            mock_embed.return_value = [[0.1] * 768] * 3  # 3 facts total

            result = await augmentation.process(ctx, mock_driver)

            # Verify memories were stored in context
            assert "memories" in result.data
            memories = result.data["memories"]

            # Check facts
            assert len(memories.entity.facts) >= 2

            # Check semantic triples
            assert len(memories.entity.semantic_triples) >= 1
            triple = memories.entity.semantic_triples[0]
            assert triple.subject_name == "User"
            assert triple.predicate == "lives in"
            assert triple.object_name == "NYC"

            # Check process attributes
            assert "project:memori-fork" in memories.process.attributes

            # Check conversation summary
            assert memories.conversation.summary == "Discussion about user preferences"

    @pytest.mark.asyncio
    async def test_no_extractor_raises_runtime_error(self):
        """Test that missing extractor raises RuntimeError."""
        from memori._config import Config
        from memori.memory.augmentation._base import AugmentationContext
        from memori.memory.augmentation.augmentations.memori._augmentation import (
            AdvancedAugmentation,
        )
        from memori.memory.augmentation.input import AugmentationInput

        config = Config()
        # Explicitly set no extractor
        config.augmentation_extractor = None

        augmentation = AdvancedAugmentation(config=config)

        payload = AugmentationInput(
            entity_id="user123",
            process_id="test-process",
            conversation_id="conv-1",
            conversation_messages=[{"role": "user", "content": "test"}],
            system_prompt=None,
        )
        ctx = AugmentationContext(payload=payload)

        mock_driver = MagicMock()
        mock_driver.conversation.conn.get_dialect.return_value = "postgresql"
        mock_driver.conversation.read.return_value = None

        with pytest.raises(RuntimeError) as exc_info:
            await augmentation.process(ctx, mock_driver)

        assert "No augmentation extractor configured" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_callable_extractor_works(self):
        """Test that a simple async callable works as extractor."""

        async def callable_extractor(payload):
            return {
                "entity": {"facts": ["from callable"], "triples": []},
                "process": {"attributes": []},
                "conversation": {"summary": None},
            }

        from memori._config import Config
        from memori.memory.augmentation._base import AugmentationContext
        from memori.memory.augmentation.augmentations.memori._augmentation import (
            AdvancedAugmentation,
        )
        from memori.memory.augmentation.input import AugmentationInput

        config = Config()
        config.augmentation_extractor = callable_extractor

        augmentation = AdvancedAugmentation(config=config)

        payload = AugmentationInput(
            entity_id="user123",
            process_id=None,
            conversation_id="conv-1",
            conversation_messages=[{"role": "user", "content": "test"}],
            system_prompt=None,
        )
        ctx = AugmentationContext(payload=payload)

        mock_driver = MagicMock()
        mock_driver.conversation.conn.get_dialect.return_value = "postgresql"
        mock_driver.conversation.read.return_value = None
        mock_driver.entity.create.return_value = 1

        with patch(
            "memori.memory.augmentation.augmentations.memori._augmentation.embed_texts_async",
            new_callable=AsyncMock,
        ) as mock_embed:
            mock_embed.return_value = [[0.1] * 768]

            result = await augmentation.process(ctx, mock_driver)

            assert "memories" in result.data
            assert "from callable" in result.data["memories"].entity.facts


class TestCLICloudCommandsRemoved:
    """Test D: CLI cloud commands removed or stubbed."""

    def test_cli_help_shows_removed_commands(self):
        """Test that CLI help shows cloud commands as removed."""
        result = subprocess.run(
            [sys.executable, "-m", "memori", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        # The CLI should run without error (exit 0 for help)
        # It should show [REMOVED] for cloud commands
        output = result.stdout + result.stderr

        # Check that cloud commands are marked as removed
        assert "[REMOVED]" in output or "REMOVED" in output.lower(), (
            f"CLI should mark cloud commands as removed. Output: {output}"
        )

    def test_quota_command_fails_with_message(self):
        """Test that quota command fails with appropriate message."""
        result = subprocess.run(
            [sys.executable, "-m", "memori", "quota"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        # Should fail (non-zero exit code)
        assert result.returncode != 0

        # Should have informative message
        output = result.stdout + result.stderr
        assert "removed" in output.lower() or "offline" in output.lower(), (
            f"Should mention removal. Output: {output}"
        )

    def test_signup_command_fails_with_message(self):
        """Test that sign-up command fails with appropriate message."""
        result = subprocess.run(
            [sys.executable, "-m", "memori", "sign-up", "test@example.com"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        # Should fail (non-zero exit code)
        assert result.returncode != 0

        # Should have informative message
        output = result.stdout + result.stderr
        assert "removed" in output.lower() or "offline" in output.lower(), (
            f"Should mention removal. Output: {output}"
        )

    def test_cockroachdb_cluster_command_fails_with_message(self):
        """Test that cockroachdb cluster command fails with appropriate message."""
        result = subprocess.run(
            [sys.executable, "-m", "memori", "cockroachdb", "cluster", "start"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        # Should fail (non-zero exit code)
        assert result.returncode != 0

        # Should have informative message
        output = result.stdout + result.stderr
        assert "removed" in output.lower() or "offline" in output.lower(), (
            f"Should mention removal. Output: {output}"
        )

    def test_setup_command_still_works(self):
        """Test that setup command is still available."""
        result = subprocess.run(
            [sys.executable, "-m", "memori"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )

        output = result.stdout + result.stderr

        # Setup should be listed and NOT marked as removed
        assert "setup" in output.lower()
        # The setup description should not contain [REMOVED]
        lines = output.split("\n")
        setup_line = next((l for l in lines if "setup" in l.lower()), "")
        assert "[REMOVED]" not in setup_line


class TestGroqExtractorContract:
    """Test the Groq extractor implementation."""

    def test_groq_extractor_import(self):
        """Test that GroqExtractor can be imported."""
        from memori.memory.augmentation.extractors import GroqExtractor

        assert GroqExtractor is not None

    def test_groq_extractor_init(self):
        """Test GroqExtractor initialization."""
        from memori.memory.augmentation.extractors import GroqExtractor

        extractor = GroqExtractor(model="llama-3.3-70b-versatile")
        assert extractor.model == "llama-3.3-70b-versatile"
        assert extractor.api_key_env == "GROQ_API_KEY"

    def test_groq_extractor_raises_without_api_key(self):
        """Test that GroqExtractor raises when API key is missing."""
        # Ensure no API key is set
        if "GROQ_API_KEY" in os.environ:
            del os.environ["GROQ_API_KEY"]

        from memori.memory.augmentation.extractors import GroqExtractor

        extractor = GroqExtractor()

        # The error should be raised when trying to get the client
        with pytest.raises(RuntimeError) as exc_info:
            extractor._get_client()

        assert "API key not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_groq_extractor_normalize_response(self):
        """Test response normalization."""
        from memori.memory.augmentation.extractors import GroqExtractor

        extractor = GroqExtractor()

        # Test with None values
        result = extractor._normalize_response(
            {
                "entity": {"facts": None, "triples": None},
                "process": {"attributes": None},
                "conversation": {"summary": None},
            }
        )

        assert result["entity"]["facts"] == []
        assert result["entity"]["triples"] == []
        assert result["process"]["attributes"] == []
        assert result["conversation"]["summary"] is None

    def test_groq_extractor_parse_json(self):
        """Test JSON parsing with markdown code blocks."""
        from memori.memory.augmentation.extractors import GroqExtractor

        extractor = GroqExtractor()

        # Test with markdown code block
        text = '''```json
{"entity": {"facts": ["test"]}}
```'''
        result = extractor._try_parse_json(text)
        assert result is not None
        assert result["entity"]["facts"] == ["test"]

        # Test plain JSON
        text = '{"entity": {"facts": ["test2"]}}'
        result = extractor._try_parse_json(text)
        assert result is not None
        assert result["entity"]["facts"] == ["test2"]

        # Test invalid JSON
        result = extractor._try_parse_json("not json at all")
        assert result is None

