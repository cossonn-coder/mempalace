"""Embedding function factory with hardware acceleration.

Returns a ChromaDB-compatible embedding function. Two backends are supported:

* **ONNX MiniLM-L6-v2 (default, dim 384)** — the historical embedder shipped
  by ChromaDB. Bound to a user-selected ONNX Runtime execution provider
  (``embedding_device``). Re-used when ``model_name`` is omitted so existing
  palaces remain readable without configuration changes.
* **Sentence-Transformers (any HF model, dim depends on model)** — opt-in
  via ``model_name="…"``. Routes through ChromaDB's
  ``SentenceTransformerEmbeddingFunction`` so any Hugging Face
  sentence-embedding model is usable (e.g.
  ``paraphrase-multilingual-mpnet-base-v2`` for dim 768 multilingual).

Both backends spoof ``name() == "default"`` so ChromaDB accepts the EF on
collections created by either path — the actual model identity is persisted
alongside the palace by the backend (see ``ChromaBackend._write_embedder_marker``),
not on the collection itself.

Supported devices (env ``MEMPALACE_EMBEDDING_DEVICE`` or ``embedding_device``
in ``~/.mempalace/config.json``) — apply to the ONNX path only:

* ``auto`` — prefer CUDA ▸ CoreML ▸ DirectML, fall back to CPU
* ``cpu`` — force CPU (the historical default)
* ``cuda`` — NVIDIA GPU via ``onnxruntime-gpu`` (``pip install mempalace[gpu]``)
* ``coreml`` — Apple Neural Engine (macOS)
* ``dml`` — DirectML (Windows / AMD / Intel GPUs)

Requesting an unavailable accelerator emits a warning and falls back to CPU
rather than hard-failing — mining must still work on a laptop without CUDA.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_PROVIDER_MAP = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
}

_DEVICE_EXTRA = {
    "cuda": "mempalace[gpu]",
    "coreml": "mempalace[coreml]",
    "dml": "mempalace[dml]",
}

_AUTO_ORDER = [
    ("CUDAExecutionProvider", "cuda"),
    ("CoreMLExecutionProvider", "coreml"),
    ("DmlExecutionProvider", "dml"),
]

_EF_CACHE: dict = {}
_WARNED: set = set()


def _resolve_providers(device: str) -> tuple[list, str]:
    """Return ``(provider_list, effective_device)`` for ``device``.

    Falls back to CPU (with a one-shot warning) when the requested
    accelerator is not compiled into the installed ``onnxruntime``.
    """
    device = (device or "auto").strip().lower()

    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except ImportError:
        return (["CPUExecutionProvider"], "cpu")

    if device == "auto":
        for provider, name in _AUTO_ORDER:
            if provider in available:
                return ([provider, "CPUExecutionProvider"], name)
        return (["CPUExecutionProvider"], "cpu")

    requested = _PROVIDER_MAP.get(device)
    if requested is None:
        if device not in _WARNED:
            logger.warning("Unknown embedding_device %r — falling back to cpu", device)
            _WARNED.add(device)
        return (["CPUExecutionProvider"], "cpu")

    preferred = requested[0]
    if preferred == "CPUExecutionProvider":
        return (requested, "cpu")

    if preferred not in available:
        if device not in _WARNED:
            extra = _DEVICE_EXTRA.get(device, "the matching mempalace extra for your device")
            logger.warning(
                "embedding_device=%r requested but %s is not installed — "
                "falling back to CPU. Install %s.",
                device,
                preferred,
                extra,
            )
            _WARNED.add(device)
        return (["CPUExecutionProvider"], "cpu")

    return (requested, device)


def _build_ef_class():
    """Subclass ``ONNXMiniLM_L6_V2`` with name ``"default"``.

    Why the rename: ChromaDB 1.5 persists the EF identity on the collection
    and rejects reads that pass a differently-named EF (``onnx_mini_lm_l6_v2``
    vs ``default``). The vectors and model are identical — only the
    ``name()`` tag differs — so spoofing the name lets one EF class serve
    palaces created with ``DefaultEmbeddingFunction`` *and* palaces we
    create ourselves, with the same GPU-capable ``preferred_providers``.
    """
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    class _MempalaceONNX(ONNXMiniLM_L6_V2):
        @staticmethod
        def name() -> str:
            return "default"

    return _MempalaceONNX


def _build_st_ef_class():
    """Subclass ``SentenceTransformerEmbeddingFunction`` with name ``"default"``.

    Why the spoof, take 2: ChromaDB's native sentence-transformers EF reports
    ``name() == "sentence_transformer"``. If we let that pass through, a
    collection created with the ONNX path (``name == "default"``) cannot be
    reopened with the ST path, and vice versa — ChromaDB rejects the EF on
    identity mismatch. Spoofing both backends to ``"default"`` lifts the
    on-collection guard; the model identity that actually matters
    (and which determines vector dimensionality) is persisted alongside the
    palace by the backend, not on the collection.
    """
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    class _MempalaceST(SentenceTransformerEmbeddingFunction):
        @staticmethod
        def name() -> str:
            return "default"

    return _MempalaceST


def get_embedding_function(
    device: Optional[str] = None,
    model_name: Optional[str] = None,
):
    """Return a cached embedding function bound to the requested device and model.

    ``device=None`` reads from :class:`MempalaceConfig.embedding_device`.
    ``model_name=None`` selects the legacy ONNX MiniLM-L6-v2 backend (dim 384,
    historical default); any other value routes through sentence-transformers
    with the named Hugging Face model. The dim of the returned vectors is
    determined by the model, not by this factory — callers that care about
    dim consistency should consult the palace-side embedder marker, not the
    EF object.

    The returned function is cached by ``(model_name, providers)`` so we only
    pay model-load cost once per process per backend.
    """
    if model_name is None:
        # Legacy ONNX MiniLM-L6 path — unchanged behaviour for backward compat.
        if device is None:
            from .config import MempalaceConfig

            device = MempalaceConfig().embedding_device

        providers, effective = _resolve_providers(device)
        cache_key = ("onnx-mini-lm-l6-v2", tuple(providers))
        cached = _EF_CACHE.get(cache_key)
        if cached is not None:
            return cached

        ef_cls = _build_ef_class()
        ef = ef_cls(preferred_providers=providers)
        _EF_CACHE[cache_key] = ef
        logger.info(
            "Embedding function initialized "
            "(device=%s providers=%s model=onnx-mini-lm-l6-v2)",
            effective,
            providers,
        )
        return ef

    # sentence-transformers path — opt-in via model_name.
    cache_key = ("st", model_name)
    cached = _EF_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        ef_cls = _build_st_ef_class()
        ef = ef_cls(model_name=model_name)
    except Exception as exc:
        # Unknown / unloadable model: fail explicitly here rather than letting
        # a dim mismatch surface deep inside ChromaDB on the next query.
        raise ValueError(
            f"Failed to instantiate sentence-transformers embedder "
            f"for model_name={model_name!r}: {exc}"
        ) from exc

    _EF_CACHE[cache_key] = ef
    logger.info(
        "Embedding function initialized "
        "(model=%s via sentence-transformers)",
        model_name,
    )
    return ef


def describe_device(device: Optional[str] = None) -> str:
    """Return a short human-readable label for the resolved device.

    Used by the miner CLI header so users can see at a glance whether GPU
    acceleration actually engaged.
    """
    if device is None:
        from .config import MempalaceConfig

        device = MempalaceConfig().embedding_device
    _, effective = _resolve_providers(device)
    return effective
