"""Exports a cross-encoder to ONNX and quantizes its weights to int8, for CPU serving. Takes under a minute; a rerun skips files on disk.

The fp32 file should score exactly as torch does, and the script checks that. int8 doesn't: judge it end to end, as Row 9 with
``--rerankers bge-reranker-v2-m3-int8``, and time it with ``scripts/query_timing.py --reranker onnx:PATH``.

    uv run --extra onnx python -u scripts/export_reranker_onnx.py
"""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from spacebridge_bench import durable
from spacebridge_bench.rerank import ONNX_DIR

PAIRS = [
    ("what is the capital of france", "Paris is the capital and largest city of France."),
    ("what is the capital of france", "Berlin is the capital of Germany."),
    ("do statins reduce heart disease risk", "Statins lower LDL cholesterol and reduce cardiovascular events in many patients."),
    ("do statins reduce heart disease risk", "The Eiffel Tower is a wrought-iron lattice tower in Paris."),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--out", type=Path, default=Path(ONNX_DIR) / "bge-reranker-v2-m3")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fp32, int8 = args.out / "model.onnx", args.out / "model_int8.onnx"

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, torch_dtype=torch.float32).eval()
    enc = tok([q for q, _ in PAIRS], [d for _, d in PAIRS], padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        reference = model(**enc).logits.reshape(-1).numpy()
    if not fp32.exists():
        axes = {"input_ids": {0: "batch", 1: "tokens"}, "attention_mask": {0: "batch", 1: "tokens"}, "logits": {0: "batch"}}
        torch.onnx.export(model, (enc["input_ids"], enc["attention_mask"]), str(fp32), input_names=["input_ids", "attention_mask"],
                          output_names=["logits"], dynamic_axes=axes, opset_version=17, dynamo=False)
    if not int8.exists():
        tmp = int8.with_name(f"{int8.name}.tmp")
        quantize_dynamic(str(fp32), str(tmp), weight_type=QuantType.QInt8)
        tmp.rename(int8)
    feed = {k: enc[k].numpy() for k in ("input_ids", "attention_mask")}
    for path in (fp32, int8):
        scores = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(["logits"], feed)[0].reshape(-1)
        durable.progress(f"{path.name}: largest difference from torch {np.abs(scores - reference).max():.3f}; {path.stat().st_size / 2**20:,.0f} MB")
    assert np.abs(ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"]).run(["logits"], feed)[0].reshape(-1) - reference).max() < 1e-2


if __name__ == "__main__":
    main()
