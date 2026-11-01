"""Embedding model registry with native prompts (design §7.1 prompt table)."""

from dataclasses import dataclass, field, replace
from typing import Optional


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    dim: int
    params_b: float
    query_template: str               # placeholders: {task}, {text}
    doc_template: str = "{body}"      # placeholders: {body} = "title\ntext", {title_or_none}, {text}
    max_seq_length: int = 512
    batch_tokens: int = 20_000        # tokens per batch once padded (measured per model); decoders like small batches
    langs: tuple[str, ...] = ("en",)
    tokenizer_kwargs: dict = field(default_factory=dict)
    config_kwargs: dict = field(default_factory=dict)
    truncate_dim: Optional[int] = None  # MRL slice: keep the first dims and re-normalize
    cache_key: str = ""                 # embedding cache directory; MRL slices share the full model's

    def format_query(self, text: str, task: str) -> str:
        return self.query_template.format(task=task, text=text)

    def format_doc(self, title: str, text: str) -> str:
        body = f"{title}\n{text}" if title else text
        return self.doc_template.format(body=body, title_or_none=title or "none", text=text)


_BGE_EN_QUERY = "Represent this sentence for searching relevant passages: {text}"

MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in [
        ModelSpec("minilm", "sentence-transformers/all-MiniLM-L6-v2", 384, 0.023, "{text}", max_seq_length=256),
        ModelSpec("bge-base-en", "BAAI/bge-base-en-v1.5", 768, 0.11, _BGE_EN_QUERY),
        ModelSpec("bge-large-en", "BAAI/bge-large-en-v1.5", 1024, 0.335, _BGE_EN_QUERY),
        ModelSpec(
            "bge-large-zh", "BAAI/bge-large-zh-v1.5", 1024, 0.326,
            "为这个句子生成表示以用于检索相关文章：{text}", langs=("zh",),
        ),
        ModelSpec(
            "me5-large", "intfloat/multilingual-e5-large", 1024, 0.56,
            "query: {text}", doc_template="passage: {body}", langs=("multi",),
        ),
        ModelSpec(
            "qwen3-0.6b", "Qwen/Qwen3-Embedding-0.6B", 1024, 0.6,
            "Instruct: {task}\nQuery:{text}", batch_tokens=5_000, langs=("multi",),
            tokenizer_kwargs={"padding_side": "left"},
        ),
        ModelSpec(
            "qwen3-4b", "Qwen/Qwen3-Embedding-4B", 2560, 4.0,
            "Instruct: {task}\nQuery:{text}", batch_tokens=5_000, langs=("multi",),
            tokenizer_kwargs={"padding_side": "left"},
        ),
        ModelSpec(
            "harrier-0.6b", "microsoft/harrier-oss-v1-0.6b", 1024, 0.6,
            "Instruct: {task}\nQuery: {text}", batch_tokens=5_000, langs=("multi",),
        ),
        ModelSpec(
            "embeddinggemma-2", "google/embeddinggemma-2", 768, 0.27,
            "task: search result | query: {text}", doc_template="title: {title_or_none} | text: {text}",
            batch_tokens=5_000, langs=("multi",), config_kwargs={"vision_config": None, "audio_config": None},  # text-only, 270M
        ),
    ]
}

# Tier 4b: Matryoshka slices of the full embeddings (the encoder caches full vectors and slices them).
for _base in ("embeddinggemma-2", "qwen3-0.6b"):
    for _d in (128, 256, 512, 768):
        _spec = MODELS[_base]
        if _d < _spec.dim:
            MODELS[f"{_base}-{_d}"] = replace(_spec, key=f"{_base}-{_d}", dim=_d, truncate_dim=_d, cache_key=_base)
