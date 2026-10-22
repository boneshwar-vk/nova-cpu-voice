"""Minimal ctypes binding for the LiteRT-LM v0.17.1 C API (raw session path).

The shared library is the one shipped in the official `litert-lm-api` 0.17.1 wheel
(liblitert-lm.dylib on macOS, liblitert-lm.so on Linux); LITERT_LIB overrides the path.
The raw session API lets the server keep one live KV cache, prefill only new text, save
checkpoints and rewind to them (speculative prefill and barge-in), which the higher-level
conversation API does not expose.
"""
from __future__ import annotations

import ctypes as C
import importlib.util
import os
import sys
import threading
import time
from pathlib import Path


def _default_lib() -> str:
    spec = importlib.util.find_spec("litert_lm")
    if spec is None or not spec.origin:
        raise RuntimeError("litert-lm-api is not installed (pip install -r requirements.txt)")
    name = "liblitert-lm.dylib" if sys.platform == "darwin" else "liblitert-lm.so"
    return str(Path(spec.origin).parent / name)


DYLIB = os.environ.get("LITERT_LIB") or _default_lib()
MODEL = os.environ.get("LITERT_MODEL", str(Path(__file__).resolve().parent / "models" / "gemma-4-E2B-it.litertlm"))

P = C.c_void_p
CB = C.CFUNCTYPE(None, C.c_void_p, C.c_void_p)

_SIGS = {
    "litert_lm_set_min_log_level": (None, [C.c_int]),
    "litert_lm_input_data_create": (P, [C.c_int, C.c_void_p, C.c_size_t]),
    "litert_lm_input_data_delete": (None, [P]),
    "litert_lm_engine_settings_create": (P, [C.c_char_p, C.c_char_p, C.c_char_p, C.c_char_p]),
    "litert_lm_engine_settings_delete": (None, [P]),
    "litert_lm_engine_settings_set_max_num_tokens": (None, [P, C.c_int]),
    "litert_lm_engine_settings_set_num_threads": (None, [P, C.c_int]),
    "litert_lm_engine_settings_set_cache_dir": (None, [P, C.c_char_p]),
    "litert_lm_engine_settings_enable_benchmark": (None, [P]),
    "litert_lm_engine_settings_set_enable_speculative_decoding": (None, [P, C.c_bool]),
    "litert_lm_engine_settings_set_use_ringbuffers_local_attention": (None, [P, C.c_bool]),
    "litert_lm_engine_settings_set_prefill_chunk_size": (None, [P, C.c_int]),
    "litert_lm_engine_settings_set_activation_data_type": (None, [P, C.c_int]),
    "litert_lm_engine_settings_set_enable_ynnpack": (None, [P, C.c_bool]),
    "litert_lm_engine_create": (P, [P]),
    "litert_lm_engine_delete": (None, [P]),
    "litert_lm_engine_tokenize": (P, [P, C.c_char_p]),
    "litert_lm_tokenize_result_get_num_tokens": (C.c_size_t, [P]),
    "litert_lm_tokenize_result_get_tokens": (C.POINTER(C.c_int), [P]),
    "litert_lm_tokenize_result_delete": (None, [P]),
    "litert_lm_engine_detokenize": (P, [P, C.POINTER(C.c_int), C.c_size_t]),
    "litert_lm_detokenize_result_get_string": (C.c_char_p, [P]),
    "litert_lm_detokenize_result_delete": (None, [P]),
    "litert_lm_sampler_params_create": (P, [C.c_int]),
    "litert_lm_sampler_params_delete": (None, [P]),
    "litert_lm_sampler_params_set_top_k": (None, [P, C.c_int]),
    "litert_lm_sampler_params_set_top_p": (None, [P, C.c_float]),
    "litert_lm_sampler_params_set_temperature": (None, [P, C.c_float]),
    "litert_lm_sampler_params_set_seed": (None, [P, C.c_int]),
    "litert_lm_session_config_create": (P, []),
    "litert_lm_session_config_delete": (None, [P]),
    "litert_lm_session_config_set_max_output_tokens": (None, [P, C.c_int]),
    "litert_lm_session_config_set_apply_prompt_template": (None, [P, C.c_bool]),
    "litert_lm_session_config_set_sampler_params": (None, [P, P]),
    "litert_lm_engine_create_session": (P, [P, P]),
    "litert_lm_session_delete": (None, [P]),
    "litert_lm_session_cancel_process": (None, [P]),
    "litert_lm_session_run_prefill": (C.c_int, [P, C.POINTER(C.c_void_p), C.c_size_t]),
    "litert_lm_session_run_decode_async": (C.c_int, [P, CB, C.c_void_p]),
    "litert_lm_session_save_checkpoint": (C.c_int, [P, C.c_char_p]),
    "litert_lm_session_rewind_to_checkpoint": (C.c_int, [P, C.c_char_p]),
    "litert_lm_session_rewind_to_step": (C.c_int, [P, C.c_int]),
    "litert_lm_session_get_benchmark_info": (P, [P]),
    "litert_lm_stream_chunk_get_text": (C.c_char_p, [P]),
    "litert_lm_stream_chunk_is_final": (C.c_bool, [P]),
    "litert_lm_stream_chunk_get_error": (C.c_char_p, [P]),
    "litert_lm_conversation_config_create": (P, []),
    "litert_lm_conversation_config_delete": (None, [P]),
    "litert_lm_conversation_config_set_session_config": (None, [P, P]),
    "litert_lm_conversation_config_set_system_message": (None, [P, C.c_char_p]),
    "litert_lm_conversation_config_set_messages": (None, [P, C.c_char_p]),
    "litert_lm_conversation_create": (P, [P, P]),
    "litert_lm_conversation_delete": (None, [P]),
    "litert_lm_conversation_render_message_to_string": (C.c_char_p, [P, C.c_char_p]),
    "litert_lm_conversation_render_preface_to_string": (C.c_char_p, [P]),
    "litert_lm_json_response_get_string": (C.c_char_p, [P]),
    "litert_lm_json_response_delete": (None, [P]),
    "litert_lm_conversation_send_message_stream": (C.c_int, [P, C.c_char_p, C.c_char_p, P, CB, C.c_void_p]),
    "litert_lm_conversation_optional_args_create": (P, []),
    "litert_lm_conversation_optional_args_delete": (None, [P]),
    "litert_lm_conversation_get_benchmark_info": (P, [P]),
    "litert_lm_benchmark_info_delete": (None, [P]),
    "litert_lm_benchmark_info_get_time_to_first_token": (C.c_double, [P]),
    "litert_lm_benchmark_info_get_num_prefill_turns": (C.c_int, [P]),
    "litert_lm_benchmark_info_get_num_decode_turns": (C.c_int, [P]),
    "litert_lm_benchmark_info_get_prefill_token_count_at": (C.c_int, [P, C.c_int]),
    "litert_lm_benchmark_info_get_decode_token_count_at": (C.c_int, [P, C.c_int]),
    "litert_lm_benchmark_info_get_prefill_tokens_per_sec_at": (C.c_double, [P, C.c_int]),
    "litert_lm_benchmark_info_get_decode_tokens_per_sec_at": (C.c_double, [P, C.c_int]),
}


def load_lib(path: str = DYLIB):
    lib = C.CDLL(path, mode=C.RTLD_GLOBAL)
    missing = []
    for name, (res, args) in _SIGS.items():
        try:
            f = getattr(lib, name)
        except AttributeError:
            missing.append(name)
            continue
        f.restype = res
        f.argtypes = args
    if missing:
        print("litert_capi: missing symbols", missing)
    return lib


L = load_lib()


def b(s):
    return None if s is None else s.encode()


class Engine:
    def __init__(self, model=MODEL, backend="cpu", max_tokens=2048, threads=None, mtp=None,
                 ringbuffer=None, cache_dir=None, log_level=None, prefill_chunk=None, act=None, ynnpack=None):
        # 4 = errors only: hides the harmless 'NPU accelerator could not be loaded' warnings (LITERT_LOG_LEVEL=3 shows them)
        L.litert_lm_set_min_log_level(int(os.environ.get("LITERT_LOG_LEVEL", "4")) if log_level is None else log_level)
        s = L.litert_lm_engine_settings_create(b(model), b(backend), None, None)
        assert s, "settings_create failed"
        L.litert_lm_engine_settings_set_max_num_tokens(s, max_tokens)
        L.litert_lm_engine_settings_enable_benchmark(s)
        if threads:
            L.litert_lm_engine_settings_set_num_threads(s, threads)
        if mtp is not None:
            L.litert_lm_engine_settings_set_enable_speculative_decoding(s, mtp)
        if ringbuffer is not None:
            L.litert_lm_engine_settings_set_use_ringbuffers_local_attention(s, ringbuffer)
        if cache_dir:
            L.litert_lm_engine_settings_set_cache_dir(s, b(cache_dir))
        if prefill_chunk:
            L.litert_lm_engine_settings_set_prefill_chunk_size(s, prefill_chunk)
        if act is not None:  # 0 fp32, 1 fp16, 2 int16, 3 int8
            L.litert_lm_engine_settings_set_activation_data_type(s, act)
        if ynnpack is not None:
            L.litert_lm_engine_settings_set_enable_ynnpack(s, ynnpack)
        t = time.perf_counter()
        self.h = L.litert_lm_engine_create(s)
        self.load_s = time.perf_counter() - t
        L.litert_lm_engine_settings_delete(s)
        assert self.h, "engine_create failed"

    def tokens(self, text: str) -> list[int]:
        r = L.litert_lm_engine_tokenize(self.h, b(text))
        n = L.litert_lm_tokenize_result_get_num_tokens(r)
        p = L.litert_lm_tokenize_result_get_tokens(r)
        out = [p[i] for i in range(n)]
        L.litert_lm_tokenize_result_delete(r)
        return out

    def ntok(self, text: str) -> int:
        return len(self.tokens(text))

    def detok(self, ids: list[int]) -> str:
        arr = (C.c_int * len(ids))(*ids)
        r = L.litert_lm_engine_detokenize(self.h, arr, len(ids))
        s = L.litert_lm_detokenize_result_get_string(r)
        L.litert_lm_detokenize_result_delete(r)
        return s.decode(errors="replace") if s else ""

    def close(self):
        if self.h:
            L.litert_lm_engine_delete(self.h)
            self.h = None


def session_config(top_k=1, temperature=1.0, top_p=1.0, seed=1, max_out=None, apply_template=False):
    sc = L.litert_lm_session_config_create()
    sp = L.litert_lm_sampler_params_create(2)  # TOP_P: the only type the CPU sampler implements (GREEDY=3 -> UNIMPLEMENTED)
    L.litert_lm_sampler_params_set_top_k(sp, top_k)
    L.litert_lm_sampler_params_set_top_p(sp, top_p)
    L.litert_lm_sampler_params_set_temperature(sp, temperature)
    L.litert_lm_sampler_params_set_seed(sp, seed)
    L.litert_lm_session_config_set_sampler_params(sc, sp)
    L.litert_lm_sampler_params_delete(sp)
    if max_out is not None:
        L.litert_lm_session_config_set_max_output_tokens(sc, max_out)
    L.litert_lm_session_config_set_apply_prompt_template(sc, apply_template)
    return sc


class Stream:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.chunks: list[tuple[float, str]] = []
        self.err = None
        self.done = threading.Event()
        self.cancel_after = None
        self.on_cancel = None

        def _cb(_data, chunk):
            now = time.perf_counter()
            err = L.litert_lm_stream_chunk_get_error(chunk)
            text = L.litert_lm_stream_chunk_get_text(chunk)
            final = L.litert_lm_stream_chunk_is_final(chunk)
            if err:
                self.err = err.decode(errors="replace")
            if text:
                self.chunks.append((now, text.decode(errors="replace")))
                if self.cancel_after and self.on_cancel and self.cancel_after(self):
                    self.cancel_after = None
                    # never re-enter the runtime from its own callback thread
                    threading.Thread(target=self.on_cancel, daemon=True).start()
            if final or err:
                self.done.set()

        self.cb = CB(_cb)

    def text(self) -> str:
        return "".join(c for _, c in self.chunks)

    @property
    def ttft(self):
        return self.chunks[0][0] - self.t0 if self.chunks else None

    def wait(self, timeout=300):
        if not self.done.wait(timeout):
            raise TimeoutError("stream timeout")
        return self


class Session:
    def __init__(self, eng: Engine, **sampler):
        self.eng = eng
        sc = session_config(**sampler)
        self.h = L.litert_lm_engine_create_session(eng.h, sc)
        L.litert_lm_session_config_delete(sc)
        assert self.h, "create_session failed"

    def prefill(self, text: str) -> float:
        raw = text.encode()
        inp = L.litert_lm_input_data_create(0, raw, len(raw))
        arr = (C.c_void_p * 1)(inp)
        t = time.perf_counter()
        rc = L.litert_lm_session_run_prefill(self.h, arr, 1)
        dt = time.perf_counter() - t
        L.litert_lm_input_data_delete(inp)
        if rc != 0:
            raise RuntimeError(f"run_prefill rc={rc}")
        return dt

    def decode(self, cancel_after=None, wait=True) -> Stream:
        st = Stream()
        st.cancel_after = cancel_after
        st.on_cancel = lambda: L.litert_lm_session_cancel_process(self.h)
        st.t0 = time.perf_counter()
        rc = L.litert_lm_session_run_decode_async(self.h, st.cb, None)
        if rc != 0:
            raise RuntimeError(f"run_decode_async rc={rc}")
        return st.wait() if wait else st

    def checkpoint(self, label: str) -> int:
        return L.litert_lm_session_save_checkpoint(self.h, b(label))

    def rewind(self, label: str) -> int:
        return L.litert_lm_session_rewind_to_checkpoint(self.h, b(label))

    def rewind_step(self, step: int) -> int:
        return L.litert_lm_session_rewind_to_step(self.h, step)

    def cancel(self):
        L.litert_lm_session_cancel_process(self.h)

    def bench(self) -> dict:
        return bench_info(L.litert_lm_session_get_benchmark_info(self.h))

    def close(self):
        if self.h:
            L.litert_lm_session_delete(self.h)
            self.h = None


def bench_info(bi) -> dict:
    if not bi:
        return {}
    out = {"ttft_s": L.litert_lm_benchmark_info_get_time_to_first_token(bi), "prefill": [], "decode": []}
    for i in range(L.litert_lm_benchmark_info_get_num_prefill_turns(bi)):
        out["prefill"].append((L.litert_lm_benchmark_info_get_prefill_token_count_at(bi, i),
                               round(L.litert_lm_benchmark_info_get_prefill_tokens_per_sec_at(bi, i), 1)))
    for i in range(L.litert_lm_benchmark_info_get_num_decode_turns(bi)):
        out["decode"].append((L.litert_lm_benchmark_info_get_decode_token_count_at(bi, i),
                              round(L.litert_lm_benchmark_info_get_decode_tokens_per_sec_at(bi, i), 2)))
    L.litert_lm_benchmark_info_delete(bi)
    return out


class Conversation:
    """Conversation API: reference output for the session's manual prompt template."""

    def __init__(self, eng: Engine, system: str | None, messages_json: str | None = None, **sampler):
        sc = session_config(apply_template=True, **sampler)
        cc = L.litert_lm_conversation_config_create()
        L.litert_lm_conversation_config_set_session_config(cc, sc)
        if system:
            L.litert_lm_conversation_config_set_system_message(cc, b(system))
        if messages_json:
            L.litert_lm_conversation_config_set_messages(cc, b(messages_json))
        self.h = L.litert_lm_conversation_create(eng.h, cc)
        L.litert_lm_conversation_config_delete(cc)
        L.litert_lm_session_config_delete(sc)
        assert self.h, "conversation_create failed"

    def render(self, message_json: str) -> str:
        r = L.litert_lm_conversation_render_message_to_string(self.h, b(message_json))
        return r.decode() if r else ""

    def preface(self) -> str:
        r = L.litert_lm_conversation_render_preface_to_string(self.h)
        return r.decode() if r else ""

    def say(self, text: str) -> Stream:
        import json
        st = Stream()
        oa = L.litert_lm_conversation_optional_args_create()
        msg = json.dumps({"role": "user", "content": [{"type": "text", "text": text}]})
        st.t0 = time.perf_counter()
        rc = L.litert_lm_conversation_send_message_stream(self.h, b(msg), None, oa, st.cb, None)
        L.litert_lm_conversation_optional_args_delete(oa)
        if rc != 0:
            raise RuntimeError(f"send_message_stream rc={rc}")
        st.wait()
        return st

    def bench(self) -> dict:
        return bench_info(L.litert_lm_conversation_get_benchmark_info(self.h))

    def close(self):
        if self.h:
            L.litert_lm_conversation_delete(self.h)
            self.h = None


_SCORE_SIGS = {
    "litert_lm_session_run_text_scoring": (P, [P, C.POINTER(C.c_char_p), C.c_size_t, C.c_bool]),
    "litert_lm_responses_get_num_candidates": (C.c_int, [P]),
    "litert_lm_responses_has_score_at": (C.c_bool, [P, C.c_int]),
    "litert_lm_responses_get_score_at": (C.c_float, [P, C.c_int]),
    "litert_lm_responses_delete": (None, [P]),
}
for _n, (_r, _a) in _SCORE_SIGS.items():
    _f = getattr(L, _n)
    _f.restype = _r
    _f.argtypes = _a


def score(sess: "Session", target: str) -> float | None:
    arr = (C.c_char_p * 1)(target.encode())
    r = L.litert_lm_session_run_text_scoring(sess.h, arr, 1, False)
    if not r:
        return None
    v = L.litert_lm_responses_get_score_at(r, 0) if L.litert_lm_responses_has_score_at(r, 0) else None
    L.litert_lm_responses_delete(r)
    return v
