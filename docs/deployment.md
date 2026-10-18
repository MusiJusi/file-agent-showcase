# Deployment and observability

## Run directly

From the repository root:

```bash
uv sync --python 3.12 --frozen
uv run streamlit run app.py
```

Use your own `.env` as described in [endpoint configuration](local_inference.md).
The UI listens on port 8501 by default.

## Docker Compose

The portable Compose configuration runs the application and Jaeger. It connects
to an existing model endpoint; it does not allocate a GPU or launch a second
answering model. Model caches and logs use named volumes.

```bash
docker compose up --build -d
```

- Application: http://localhost:8501
- Jaeger traces: http://localhost:16686

The app container defaults to
`http://host.docker.internal:18000/v1`. Override `DOCKER_LLM_BASE_URL` in
`.env` for a different address. A separate VLM's `VLM_BASE_URL` must also be
reachable from inside the container.

On Linux, a service bound only to the host's loopback interface may not be
reachable through the bridge gateway. Use a reachable endpoint address or run
the application directly alongside the SSH tunnel. The Compose file exposes
the UI and trace ports only on the host's loopback interface.

```bash
docker compose logs -f app
docker compose down
```

Named volumes survive `docker compose down`; no machine-specific mount paths
are required. There is no model server dependency to start accidentally.

## Tracing a directly launched app

```bash
docker compose up -d jaeger
uv run streamlit run app.py
```

By default the application sends OTLP/gRPC traces to `localhost:4317`.
Compose overrides this to `jaeger:4317` inside its network.
The service name is `file-agent-pipeline`.

Trace spans cover parsing, chunking, indexing, retrieval, and answering.
When Jaeger is absent, background export can log connection failures;
answering does not require the collector.

## Optional Langfuse export

Set all three values in the application's local configuration:

```dotenv
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=
```

Supplying valid values enables the additional OTLP/HTTP exporter. Leaving all
three blank disables it. A partial configuration is rejected with a clear error.
Keep these values in your local `.env`, never in documentation or screenshots.
