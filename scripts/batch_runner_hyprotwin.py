#!/usr/bin/env python3
"""Batch runner adapted for HyProTwin WSV evaluation.

Usage:
    uv run python scripts/batch_runner_hyprotwin.py \
        --config configs/hyprotwin.yaml \
        --output results/hyprotwin/
"""

import argparse
import json
import logging
import os
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from tqdm import tqdm
from arag import LLMClient, BaseAgent, ToolRegistry, Config
from arag.tools.keyword_search import KeywordSearchTool
from arag.tools.read_document import ReadDocumentTool
from arag.tools.metadata_filter import MetadataFilterTool

logging.basicConfig(level=logging.ERROR)


class HyProTwinBatchRunner:
    """Batch runner for HyProTwin WSV evaluation."""

    def __init__(
        self,
        config: Config,
        output_dir: str,
        limit: int = None,
        num_workers: int = 5,
        verbose: bool = False,
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self.num_workers = num_workers
        self.verbose = verbose

        self.predictions_file = self.output_dir / "predictions.jsonl"
        self.write_lock = Lock()

        # Load questions
        questions_file = config.get("data.questions_file", "data/wsv/questions.json")
        with open(questions_file, "r", encoding="utf-8") as f:
            self.questions = json.load(f)
        if self.limit:
            self.questions = self.questions[: self.limit]

        # Load system prompt
        prompt_file = config.get("agent.system_prompt", "src/arag/agent/prompts/hyprotwin.txt")
        self._system_prompt = Path(prompt_file).read_text(encoding="utf-8")

        # Initialize shared tools
        self._shared_tools = self._init_tools()

    def _init_tools(self) -> ToolRegistry:
        data_config = self.config.get("data", {})
        chunks_file = data_config.get("chunks_file", "data/wsv/chunks.json")
        registry_file = data_config.get("document_registry", "data/wsv/document_registry.json")

        tools = ToolRegistry()
        tools.register(KeywordSearchTool(chunks_file=chunks_file))
        tools.register(ReadDocumentTool(registry_file=registry_file))
        tools.register(MetadataFilterTool(registry_file=registry_file))

        # Qdrant semantic search (optional)
        qdrant_config = self.config.get("qdrant", {})
        if qdrant_config.get("host"):
            try:
                from qdrant_client import QdrantClient
                from arag.tools.semantic_search_qdrant import QdrantSemanticSearchTool
                import litellm

                client = QdrantClient(
                    host=qdrant_config["host"],
                    port=qdrant_config.get("port", 6333),
                )
                # Test connection
                client.get_collections()

                llm_config = self.config.get("llm", {})
                embedding_config = self.config.get("embedding", {})
                vertex_kwargs = {}
                if llm_config.get("vertex_project"):
                    vertex_kwargs["vertex_project"] = llm_config["vertex_project"]
                if llm_config.get("vertex_location"):
                    vertex_kwargs["vertex_location"] = llm_config["vertex_location"]
                if llm_config.get("vertex_credentials"):
                    vertex_kwargs["vertex_credentials"] = llm_config["vertex_credentials"]

                def embed_fn(text: str) -> list[float]:
                    resp = litellm.embedding(
                        model=embedding_config.get("model", "text-embedding-ada-002"),
                        input=[text],
                        **vertex_kwargs,
                    )
                    return resp.data[0]["embedding"]

                tools.register(
                    QdrantSemanticSearchTool(
                        qdrant_client=client,
                        collection_name=qdrant_config.get("collection", "hyprotwin"),
                        embedding_fn=embed_fn,
                    )
                )
                print("Qdrant semantic search enabled.")
            except Exception as e:
                print(f"Qdrant semantic search disabled: {e}")

        print(f"Registered tools: {tools.list_tools()}")
        return tools

    def _create_agent(self) -> BaseAgent:
        llm_config = self.config.get("llm", {})
        client = LLMClient(
            model=llm_config.get("model"),
            vertex_project=llm_config.get("vertex_project"),
            vertex_location=llm_config.get("vertex_location"),
            vertex_credentials=llm_config.get("vertex_credentials"),
            temperature=llm_config.get("temperature", 0.0),
            max_tokens=llm_config.get("max_tokens", 16384),
        )
        agent_config = self.config.get("agent", {})
        return BaseAgent(
            llm_client=client,
            tools=self._shared_tools,
            system_prompt=self._system_prompt,
            max_loops=agent_config.get("max_loops", 15),
            max_token_budget=agent_config.get("max_token_budget", 200000),
            verbose=self.verbose,
        )

    def _load_completed_qids(self) -> set:
        completed = set()
        if not self.predictions_file.exists():
            return completed
        with open(self.predictions_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    qid = data.get("qid") or data.get("id")
                    if qid and "pred_answer" in data:
                        completed.add(qid)
                except json.JSONDecodeError:
                    continue
        return completed

    def _append_prediction(self, prediction: Dict[str, Any]):
        with self.write_lock:
            with open(self.predictions_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    def _process_one(self, item: Dict[str, Any], agent: BaseAgent) -> Dict[str, Any]:
        qid = item.get("id") or item.get("qid")
        question = item["question"]
        gold_answer = item.get("answer", "")

        try:
            result = agent.run(question)
            return {
                "qid": qid,
                "question": question,
                "gold_answer": gold_answer,
                "pred_answer": result["answer"],
                "trajectory": result["trajectory"],
                "total_cost": result["total_cost"],
                "loops": result["loops"],
                "metadata": item.get("metadata", {}),
                **{k: v for k, v in result.items()
                   if k not in ("answer", "trajectory", "total_cost", "loops")},
            }
        except Exception as e:
            return {
                "qid": qid,
                "question": question,
                "gold_answer": gold_answer,
                "pred_answer": f"Error: {e}",
                "trajectory": [],
                "total_cost": 0,
                "loops": 0,
                "metadata": item.get("metadata", {}),
                "error": str(e),
            }

    def run(self):
        completed_qids = self._load_completed_qids()
        pending = [q for q in self.questions if (q.get("id") or q.get("qid")) not in completed_qids]

        print(f"Total: {len(self.questions)} | Completed: {len(completed_qids)} | Pending: {len(pending)}")

        if not pending:
            print("All questions completed!")
            return

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            for item in pending:
                agent = self._create_agent()
                future = executor.submit(self._process_one, item, agent)
                futures[future] = item.get("id") or item.get("qid")

            with tqdm(total=len(pending), desc="Processing") as pbar:
                for future in as_completed(futures):
                    qid = futures[future]
                    try:
                        result = future.result()
                        self._append_prediction(result)
                    except Exception as e:
                        print(f"Error {qid}: {e}")
                    pbar.update(1)

        print(f"Results saved to: {self.predictions_file}")


def main():
    parser = argparse.ArgumentParser(description="HyProTwin A-RAG Batch Runner")
    parser.add_argument("--config", "-c", default="configs/hyprotwin.yaml")
    parser.add_argument("--output", "-o", default="results/hyprotwin/")
    parser.add_argument("--limit", "-l", type=int, default=None)
    parser.add_argument("--workers", "-w", type=int, default=5)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    runner = HyProTwinBatchRunner(
        config=config,
        output_dir=args.output,
        limit=args.limit,
        num_workers=args.workers,
        verbose=args.verbose,
    )
    runner.run()


if __name__ == "__main__":
    main()
