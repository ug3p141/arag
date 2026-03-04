#!/usr/bin/env python3
"""Upload WSV chunks to Qdrant with embeddings and metadata payloads.

Usage:
    uv run python scripts/upload_to_qdrant.py \
        --config configs/hyprotwin.yaml
"""

import argparse
import json

import litellm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

from arag import Config


def main():
    parser = argparse.ArgumentParser(description="Upload chunks to Qdrant")
    parser.add_argument("--config", "-c", default="configs/hyprotwin.yaml")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--recreate", action="store_true", help="Recreate collection")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    data_config = config.get("data", {})
    qdrant_config = config.get("qdrant", {})
    embedding_config = config.get("embedding", {})
    llm_config = config.get("llm", {})

    # Load chunks
    chunks_file = data_config.get("chunks_file", "data/wsv/chunks.json")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    # Connect to Qdrant
    client = QdrantClient(
        host=qdrant_config.get("host", "localhost"),
        port=qdrant_config.get("port", 6333),
    )
    collection = qdrant_config.get("collection", "hyprotwin")

    # Build vertex kwargs for embedding calls
    model = embedding_config.get("model", "text-embedding-ada-002")
    vertex_kwargs = {}
    if llm_config.get("vertex_project"):
        vertex_kwargs["vertex_project"] = llm_config["vertex_project"]
    if llm_config.get("vertex_location"):
        vertex_kwargs["vertex_location"] = llm_config["vertex_location"]
    if llm_config.get("vertex_credentials"):
        vertex_kwargs["vertex_credentials"] = llm_config["vertex_credentials"]

    # Get embedding dimension from a test call
    test_resp = litellm.embedding(model=model, input=["test"], **vertex_kwargs)
    dim = len(test_resp.data[0]["embedding"])
    print(f"Embedding model: {model}, dimension: {dim}")

    # Create/recreate collection
    if args.recreate:
        client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"Recreated collection '{collection}'")
    elif not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"Created collection '{collection}'")

    # Upload in batches
    for i in tqdm(range(0, len(chunks), args.batch_size), desc="Uploading"):
        batch = chunks[i : i + args.batch_size]
        texts = [c["text"] for c in batch]

        resp = litellm.embedding(model=model, input=texts, **vertex_kwargs)
        vectors = [d["embedding"] for d in resp.data]

        points = []
        for j, (chunk, vector) in enumerate(zip(batch, vectors)):
            points.append(
                PointStruct(
                    id=i + j,
                    vector=vector,
                    payload={
                        "chunk_id": chunk["id"],
                        "text": chunk["text"],
                        "document_id": chunk.get("document_id", ""),
                        "structure_name": chunk.get("structure_name", ""),
                        "structure_id": chunk.get("structure_id", ""),
                        "inspection_year": chunk.get("inspection_year"),
                        "report_type": chunk.get("report_type", ""),
                        "doc_type": chunk.get("doc_type", ""),
                    },
                )
            )

        client.upsert(collection_name=collection, points=points)

    print(f"Uploaded {len(chunks)} chunks to '{collection}'")


if __name__ == "__main__":
    main()
