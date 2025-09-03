# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import openai
import pytest
import pytest_asyncio

from ...utils import RemoteOpenAIServer

# any model with a chat template should work here
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mpatch = MonkeyPatch()
    yield mpatch
    mpatch.undo()


@pytest.fixture(scope="module", params=[True])
def server(request, monkeypatch_module):
    use_v1 = request.param
    monkeypatch_module.setenv('VLLM_USE_V1', '1' if use_v1 else '0')

    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--enforce-eager",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_chat_completion_with_stats(client: openai.AsyncOpenAI):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": "what is 1+1?"
    }]

    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=5,
        temperature=0.0,
    )

    assert hasattr(chat_completion, 'stats')
    assert chat_completion.stats is not None


@pytest.mark.asyncio
async def test_chat_completion_with_orca_header(server: RemoteOpenAIServer):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": "what is 1+1?"
    }]

    client = openai.OpenAI(
        api_key="EMPTY",
        base_url=f"http://localhost:{server.port}/v1",
    )

    chat_completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=5,
        temperature=0.0,
        extra_headers={"endpoint-load-metrics-format": "TEXT"},
    )

    assert "endpoint-load-metrics" in chat_completion._headers
