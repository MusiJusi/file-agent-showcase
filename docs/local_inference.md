# Connect an answering model

The application calls an OpenAI-compatible chat endpoint through its LLM adapter.
It does not start a model server. Use the exact model identifier exposed by your
server; local endpoint keys may be blank when the server does not require one.

## Recommended project profile

The example configuration uses Qwen3.5-27B for answering and visual analysis.
The embedding and reranking models run locally, independently of that endpoint.

```dotenv
LLM_BACKEND=local
LOCAL_LLM_BASE_URL=http://127.0.0.1:18000/v1
LOCAL_LLM_MODEL=Qwen/Qwen3.5-27B
LOCAL_LLM_API_KEY=
RAG_MODE=tool_agent
RAG_MAX_TOOL_ROUNDS=8
RAG_HISTORY_TURNS=6
RAG_MAX_CONTEXT_TOKENS=90000
```

The context budget is an estimate of input tokens, not a request to extend the
server's context window. Lower it for smaller-context deployments and leave room
for tool schemas and answer tokens.

Check that the server is reachable:

```bash
curl http://127.0.0.1:18000/v1/models
```

On Windows PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:18000/v1/models
```

Copy a returned model ID into `LOCAL_LLM_MODEL`. A successful models response
confirms connectivity; it does not prove tool-calling or image support.

## Tool calling and vision

Agent mode requires native OpenAI-compatible `tools`, `tool_choice`,
assistant `tool_calls`, and tool-result messages. JSON text that resembles a
tool call is not sufficient. Enable the tool parser/chat template required by
your chosen serving system and model.

`VLM_BACKEND=llm` reuses the local answering endpoint for images.
The server must accept image content. For a separate visual endpoint, select
`VLM_BACKEND=openai` and set `VLM_BASE_URL`, `VLM_MODEL`, and optionally
`VLM_API_KEY`. Select `VLM_BACKEND=off` for text-only use.

If supported by the server, `LOCAL_LLM_ENABLE_THINKING=false` requests direct
answers. Leaving it unset retains the serving configuration.
Changing this value can change both latency and answer quality.

## Remote model through SSH

A tunnel can expose a remote model port on your own machine:

```bash
ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
  -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18000:MODEL_HOST:8000 USER@SSH_HOST
```

Replace the uppercase placeholders with your deployment values and keep the
tunnel alive while generating answers. The application still uses
`http://127.0.0.1:18000/v1`. No SSH host or user is embedded in the code.

## Cloud backend

To use Yandex AI Studio, set `LLM_BACKEND=yandex` and configure
`YANDEX_API_KEY`, `YANDEX_FOLDER_ID`, `YANDEX_MODEL`, and `YANDEX_BASE_URL`.
The model identifier and available tool/vision capabilities depend on your
deployment.

Visual reuse via `VLM_BACKEND=llm` reads the **local** endpoint settings.
For a cloud answering model, configure a separate `openai` VLM endpoint or use
`VLM_BACKEND=off`.

## Resource expectations

Qwen3.5-27B can run on a separate inference server. This application does not need
to load its weights. The two Qwen3 0.6B retrieval models, PDF layout, and OCR
models can consume local memory and GPU resources. First-use downloads and
document parsing can dominate the first upload.

For Docker, use the [container endpoint settings](deployment.md) because
`127.0.0.1` inside a container refers to that container.
