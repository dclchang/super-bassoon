# Super Bassoon

AI-powered receipt extraction and semantic search for PaperlessNGX.

## Overview

Super Bassoon extracts structured data from receipts using LLMs (via LiteLLM), stores embeddings in Qdrant, and enables semantic search queries like "How much did I pay at VicRoads?".

## Features

**Privacy first**: All processing is done locally via your own LiteLLM instance. Documents are not sent to third-party cloud APIs.

- **Document Retrieval** - Syncs documents from PaperlessNGX
- **LLM Extraction** - Extracts structured data (vendor, amount, date, etc.) from receipts
- **Quality Review** - Scores extractions for accuracy
- **Vector Embedding** - Generates embeddings for semantic search
- **Semantic Search** - Query receipts using natural language

## Requirements

- Python 3.9+
- [PaperlessNGX](https://docs.paperless-ngx.com/) instance
- [LiteLLM](https://litellm.ai/) proxy (local or cloud)
- [Qdrant](https://qdrant.tech/) vector database
- 1Password CLI (for secrets)

## Installation

```bash
pip install -e .
```

## Configuration

Set these secrets via 1Password:
- `op://homelab/paperless-api-token/credential`
- `op://homelab/litellm-virtual-key-for-rag-app/credential`
- `op://homelab/qdrant-namespace/credential`

## Usage

### Retrieve documents from PaperlessNGX
```bash
python -m super_bassoon.retriever
```

### Extract and embed receipts
```bash
python -m super_bassoon.embedder
```

### Query receipts semantically
```bash
python -m super_bassoon.querier
```

## Architecture

- **PaperlessNgx** - Client for PaperlessNGX API (async)
- **LlmProxy** - LiteLLM wrapper with concurrency control (async)
- **Retriever** - Syncs documents from PaperlessNGX
- **Embedder** - Extracts, reviews, and embeds documents
- **VectorDb** - Qdrant wrapper for vector storage
- **Querier** - Semantic search over embedded documents

## License

See [LICENSE](LICENSE) for details.
