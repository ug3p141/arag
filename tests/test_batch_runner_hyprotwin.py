"""Tests for HyProTwin batch runner."""

from arag import Config


def test_config_loads_hyprotwin():
    config = Config.from_yaml("configs/hyprotwin.yaml")
    assert config.get("llm.model") == "vertex_ai/claude-sonnet-4-5-20250929"
    assert config.get("agent.max_loops") == 15
    assert config.get("data.document_registry") == "data/wsv/document_registry.json"
