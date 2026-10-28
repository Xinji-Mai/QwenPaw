# -*- coding: utf-8 -*-
"""The local chat transport feeds native approval and ReMe consumers."""
# pylint: disable=protected-access

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI
from agentscope.classifier import BinaryQuestion

from qwenpaw.config.config import AgentProfileConfig, ModelSlotConfig
from qwenpaw.providers import ProviderManager
from qwenpaw.providers.ollama_provider import OllamaProvider
from qwenpaw.providers.services.qwenpaw import QwenPawProvider
from qwenpaw.security import decision_model


@pytest.fixture(name="local", params=[QwenPawProvider, OllamaProvider])
def local_fixture(request, monkeypatch):
    # pylint: disable=too-many-statements
    cls = request.param
    provider = cls(
        id="ollama" if cls is OllamaProvider else "qwenpaw-local",
        name="Local fixture",
        base_url="http://127.0.0.1:9000"
        + ("" if cls is OllamaProvider else "/v1"),
        models=[
            {
                "id": "blink-fixture",
                "name": "Blink fixture",
                "model_type": "decision",
            },
        ],
    )
    state = SimpleNamespace(mode="ok", requests=[], clients=[])

    def handler(req):
        # pylint: disable=too-many-branches
        state.requests.append(req)
        assert req.url.path == "/v1/chat/completions"
        body = json.loads(req.content)
        assert body["model"] == "blink-fixture"
        assert body["stream"] is False
        assert body["temperature"] == 0
        assert body["response_format"]["type"] == "json_schema"
        evidence = json.loads(body["messages"][1]["content"])
        if state.mode == "http_error":
            return httpx.Response(503, json={"error": {"message": "Loading"}})
        if state.mode == "timeout":
            raise httpx.ReadTimeout("timeout", request=req)
        answers = {}
        for name, question in evidence["questions"].items():
            if question["type"] == "binary":
                answers[name] = {"probability": 0.001}
            else:
                options = list(question["criteria"])
                selected = "routine" if "routine" in options else options[0]
                probabilities = dict.fromkeys(options, 0.0)
                probabilities[selected] = 1.0
                answers[name] = {
                    "choice": selected,
                    "confidence": 0.99,
                    "probabilities": probabilities,
                }
        if "impact" in answers:
            if state.mode == "missing_confidence":
                answers["impact"].pop("confidence")
            if state.mode == "low_confidence":
                answers["impact"]["confidence"] = 0.1
            if state.mode == "missing_probability":
                answers["impact"]["probabilities"].pop("risky")
            if state.mode == "bad_sum":
                answers["impact"]["probabilities"]["risky"] = 0.5
            if state.mode == "bad_choice":
                answers["impact"]["choice"] = "risky"
            if state.mode == "risk":
                answers["destructive"]["probability"] = 0.9
            if state.mode == "boolean":
                answers["destructive"]["probability"] = False
            if state.mode == "nan":
                answers["destructive"]["probability"] = float("nan")
        content = json.dumps({"answers": answers})
        if state.mode == "bad_json":
            content = "I think it is safe"
        if state.mode == "duplicate":
            content = '{"answers":{},"answers":' + json.dumps(answers) + "}"
        return httpx.Response(
            200,
            json={
                "id": "fixture",
                "object": "chat.completion",
                "created": 0,
                "model": "blink-fixture",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": (
                            "length" if state.mode == "truncated" else "stop"
                        ),
                    },
                ],
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 20,
                    "total_tokens": 60,
                },
            },
        )

    def make_client(self, timeout=5):
        base = (
            self._openai_compatible_base_url()
            if cls is OllamaProvider
            else self.base_url
        )
        client = AsyncOpenAI(
            api_key="fixture",
            base_url=base,
            timeout=timeout,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(handler),
            ),
        )
        state.clients.append(client)
        return client

    monkeypatch.setattr(cls, "_client", make_client)
    slot = ModelSlotConfig(provider_id=provider.id, model="blink-fixture")
    manager = SimpleNamespace(
        default_models={"decision": slot},
        get_provider=lambda _: provider,
    )
    monkeypatch.setattr(ProviderManager, "get_instance", lambda: manager)
    state.provider = provider
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,needs_approval",
    [
        ("ok", False),
        ("low_confidence", True),
        ("risk", True),
        ("missing_confidence", True),
        ("missing_probability", True),
        ("bad_sum", True),
        ("bad_choice", True),
        ("boolean", True),
        ("nan", True),
        ("bad_json", True),
        ("duplicate", True),
        ("truncated", True),
        ("http_error", True),
        ("timeout", True),
    ],
)
async def test_native_approval_uses_local_selection_and_fails_closed(
    local,
    mode,
    needs_approval,
):
    local.mode = mode
    result = await decision_model.assess_tool_call(
        "write_file",
        "/tmp/synthetic.txt",
        operation="write",
        baseline_ask=True,
        params={"content": "PRIVATE_VALUE_NOT_SENT"},
    )
    assert result.needs_approval is needs_approval
    assert len(local.requests) == 1
    assert "PRIVATE_VALUE_NOT_SENT" not in local.requests[0].content.decode()
    assert all(client.is_closed() for client in local.clients)


@pytest.mark.asyncio
async def test_factory_and_reme_keep_native_answer_interface(local):
    from qwenpaw.agents.model_factory import create_decision_model_async
    from reme.components.as_decision_model import AgentScopeAsDecisionModel

    model = await create_decision_model_async(
        AgentProfileConfig(id="fixture", name="Fixture"),
    )
    component = AgentScopeAsDecisionModel()
    component.model = model
    try:
        response = await component(
            state="Synthetic tool result",
            questions={
                "keep": {"type": "binary", "instructions": "Relevant?"},
            },
        )
        assert response.content["keep"].probability == 0.001
        assert response.usage.input_tokens == 40
    finally:
        await component.close()
        await model.client.aclose()
    assert all(client.is_closed() for client in local.clients)


@pytest.mark.asyncio
async def test_chat_model_is_not_silently_used_as_decision_model(local):
    local.provider.models[0].model_type = "chat"
    result = await decision_model.assess_tool_call(
        "write_file",
        "/tmp/example",
        operation="write",
    )
    assert result.needs_approval
    assert not local.requests


@pytest.mark.asyncio
async def test_cancelled_request_closes_client(local, monkeypatch):
    from qwenpaw.providers.local_decision_model import LocalDecisionModel

    async def cancel(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(LocalDecisionModel, "__call__", cancel)
    with pytest.raises(asyncio.CancelledError):
        await local.provider.evaluate(
            model="blink-fixture",
            state="x",
            questions={"q": BinaryQuestion(instructions="Relevant?")},
        )
    assert all(client.is_closed() for client in local.clients)
