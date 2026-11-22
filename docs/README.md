# Documentation

Start with the [project overview](../README.md) or choose a guide.

| Topic | Guide |
| --- | --- |
| How the system fits together | [Architecture and module map](architecture.md) |
| How to connect an answering model | [Local and cloud endpoints](local_inference.md) |
| How to run Docker or inspect traces | [Deployment and observability](deployment.md) |
| What the reported scores measure | [Evaluation protocol and archived result](evaluation.md) |
| How to try safe sample files | [Example documents](../examples/README.md) |
| How to generate a Hugging Face run | [HF generation](hf_dataset_generation.md) |
| How to run DocBench | [DocBench generation](docbench_generation.md) |
| Who owned the RAG logic and how the team contributed | [Project background and RAG ownership](project_background.md) |
| How to make a change | [Contributing](../CONTRIBUTING.md) |

## Project scope and evaluation

File Agent was developed as a team project; **MusiJusi was responsible for the
RAG logic**. See [project background](project_background.md) for the module map
and attribution.

The golden set contains **100 files and 400 questions**. The published metrics
cover **125 selected questions**; use the [evaluation note](evaluation.md) and
[main README snapshot](../README.md#evaluation-snapshot) for the scores and protocol.

## Research notes

[Parsing and chunking](parsing_and_chunking.md) and
[retrieval](retrieval.md) retain the shared baseline's detailed engineering notes,
ablations, and historical measurements. Their experiment labels and judge
protocols belong to those experiments. Use the
[current evaluation note](evaluation.md) when citing this edition's archived result.

## Configuration

[The application example](../.env.example) is a commented reference for parsing,
retrieval, the agent, generation, and tracing.
[The evaluator example](../eval_pipeline/.env.example) configures the independent
judge. Copy either file to `.env` in its own project directory and supply your
own endpoint and credentials.

Code defaults remain backward-compatible. An example file is an explicit launch
profile and can intentionally select different values.
