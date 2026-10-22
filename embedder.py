"""CPU-only sentence embedder for the memory store: all-MiniLM-L6-v2 int8 via ONNX Runtime.

About a millisecond per query at 2 threads on a laptop CPU. Mean pooling over the attention
mask, L2-normalised 384-d output. The upstream tokenizer.json has fixed padding to 128 tokens;
it is disabled here so short queries stay cheap and pooling stays exact.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("EMBED_MODEL_DIR", HERE / "models" / "embed" / "minilm"))
MODEL_FILE = os.environ.get("EMBED_MODEL_FILE", "model_quantized.onnx")
EMBED_THREADS = int(os.environ.get("EMBED_THREADS", "2"))
MODEL_NAME = f"minilm-l6-v2/{MODEL_FILE}"
DIM = 384
MAX_LEN = 256


class Embedder:
    def __init__(self, threads: int = EMBED_THREADS) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        t0 = time.perf_counter()
        self.tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        self.tok.enable_truncation(MAX_LEN)
        self.tok.no_padding()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")  # no busy-wait between calls
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(MODEL_DIR / MODEL_FILE), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.encode(["warm up"])
        self.load_ms = (time.perf_counter() - t0) * 1000.0
        self._lock = threading.Lock()

    def encode(self, texts: list[str]) -> np.ndarray:
        encs = self.tok.encode_batch(texts)
        L = max(len(e.ids) for e in encs)
        ids = np.zeros((len(encs), L), np.int64)
        mask = np.zeros_like(ids)
        for i, e in enumerate(encs):
            ids[i, :len(e.ids)] = e.ids
            mask[i, :len(e.ids)] = 1
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        lock = getattr(self, "_lock", None)
        if lock is not None:
            with lock:
                out = self.sess.run(None, feed)[0]
        else:
            out = self.sess.run(None, feed)[0]
        m = mask[..., None].astype(np.float32)
        v = (out * m).sum(1) / np.maximum(m.sum(1), 1e-9)
        return (v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)).astype(np.float32)


_inst: Embedder | None = None
_inst_lock = threading.Lock()


def get() -> Embedder:
    global _inst
    if _inst is None:
        with _inst_lock:
            if _inst is None:
                _inst = Embedder()
    return _inst


def available() -> bool:
    return (MODEL_DIR / MODEL_FILE).exists() and (MODEL_DIR / "tokenizer.json").exists()
