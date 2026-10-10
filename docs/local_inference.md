# Local LLM inference

The application does not import vLLM or SGLang directly. `OpenAILLMClient`
uses the official OpenAI Python SDK to call a local OpenAI-compatible HTTP
endpoint. A blank `LOCAL_LLM_API_KEY` is supported for servers without
authentication.

`RAG_MODE=tool_agent` additionally requires the server/model pair to implement
OpenAI-compatible native tool calling (`tools`, `tool_choice`, assistant
`tool_calls`, and tool-result messages). A server that only returns tool-like
JSON as plain text is not sufficient. For Qwen reasoning models, disabling
thinking for tool calls is often more predictable:

```env
RAG_MODE=tool_agent
RAG_MAX_TOOL_ROUNDS=4
RAG_MAX_CONTEXT_TOKENS=90000
LOCAL_LLM_ENABLE_THINKING=false
```

`RAG_MAX_CONTEXT_TOKENS` is a conservative input budget used to compact old and
verbose tool observations before a model request. It does not truncate the source
passages stored in the returned `RAGResponse`; it only bounds the model-facing trace.

## vLLM

Example:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

Project configuration:

```env
LLM_BACKEND=local
LOCAL_LLM_BASE_URL=http://localhost:8000/v1
LOCAL_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
LOCAL_LLM_API_KEY=
```

## SGLang

Example:

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 30000
```

Project configuration:

```env
LLM_BACKEND=local
LOCAL_LLM_BASE_URL=http://localhost:30000/v1
LOCAL_LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
LOCAL_LLM_API_KEY=
```
