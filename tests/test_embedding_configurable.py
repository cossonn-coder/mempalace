"""Tests for the configurable embedder factory and palace-side persistence.

Covers the four scenarios called out in the T-Mempalace-Patch brief:

1. ``test_embedding_factory_default`` — no ``model_name`` returns the legacy
   ONNX MiniLM-L6 EF (dim 384) with the spoofed ``"default"`` name.
2. ``test_embedding_factory_mpnet`` — an explicit
   ``paraphrase-multilingual-mpnet-base-v2`` returns a sentence-transformers
   EF whose vectors are 768-dim.
3. ``test_palace_create_with_custom_embedder`` — creating a palace with the
   ``embedding_model`` option writes the marker file, and a fresh backend
   that reopens the same palace transparently re-instantiates mpnet.
4. ``test_palace_backward_compat`` — a palace created without
   ``embedding_model`` writes no marker and continues to use MiniLM 384.
"""

from __future__ import annotations

import json
import os

import pytest

from mempalace.backends.base import PalaceRef
from mempalace.backends.chroma import (
    _EMBEDDER_MARKER_FILENAME,
    ChromaBackend,
)
from mempalace.embedding import _EF_CACHE, get_embedding_function


_MPNET = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


@pytest.fixture(autouse=True)
def _clear_ef_cache():
    """Ensure each test sees a fresh factory cache."""
    _EF_CACHE.clear()
    yield
    _EF_CACHE.clear()


def test_embedding_factory_default():
    """Without ``model_name`` we get the legacy ONNX MiniLM EF (name spoofed)."""
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    ef = get_embedding_function()
    assert ef.name() == "default"
    assert isinstance(ef, ONNXMiniLM_L6_V2)


def test_embedding_factory_mpnet():
    """With mpnet, the EF is a sentence-transformers backend producing 768-dim
    vectors and still spoofs ``name() == "default"`` for ChromaDB compat."""
    from chromadb.utils.embedding_functions import (
        SentenceTransformerEmbeddingFunction,
    )

    ef = get_embedding_function(model_name=_MPNET)
    assert ef.name() == "default"
    assert isinstance(ef, SentenceTransformerEmbeddingFunction)

    vectors = ef(["bonjour le monde"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 768


def test_palace_create_with_custom_embedder(tmp_path):
    """Create a palace with mpnet, close, reopen with a fresh backend, and
    verify the marker is written and the EF is automatically re-instantiated
    on reopen (read returns 768-dim vectors)."""
    palace_path = str(tmp_path / "palace_mpnet")
    palace = PalaceRef(id="t-mpnet", local_path=palace_path)

    backend = ChromaBackend()
    try:
        coll = backend.get_collection(
            palace=palace,
            collection_name="mempalace_drawers",
            create=True,
            options={"embedding_model": _MPNET},
        )
        coll.add(
            documents=["bonjour le monde"],
            ids=["doc-1"],
            metadatas=[{"lang": "fr"}],
        )
    finally:
        backend.close_palace(palace)

    marker_path = os.path.join(palace_path, _EMBEDDER_MARKER_FILENAME)
    assert os.path.isfile(marker_path), "marker file should be written on create"
    with open(marker_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["model"] == _MPNET
    assert payload["version"] == 1

    # Fresh backend = fresh client cache; the EF must be rebuilt from the
    # marker alone (the caller does NOT pass ``embedding_model`` on reopen).
    backend2 = ChromaBackend()
    try:
        coll2 = backend2.get_collection(
            palace=palace,
            collection_name="mempalace_drawers",
            create=False,
        )
        result = coll2.query(
            query_texts=["salut tout le monde"],
            n_results=1,
            include=["embeddings"],
        )
        assert len(result.ids[0]) == 1
        assert result.ids[0][0] == "doc-1"
        # Vectors round-trip in 768 dim — confirms mpnet was re-instantiated.
        assert result.embeddings is not None
        assert len(result.embeddings[0][0]) == 768
    finally:
        backend2.close_palace(palace)


def test_palace_backward_compat(tmp_path):
    """A palace created without ``embedding_model`` writes no marker and
    behaves exactly as a pre-patch palace would: MiniLM 384."""
    palace_path = str(tmp_path / "palace_legacy")
    palace = PalaceRef(id="t-legacy", local_path=palace_path)

    backend = ChromaBackend()
    try:
        coll = backend.get_collection(
            palace=palace,
            collection_name="mempalace_drawers",
            create=True,
        )
        coll.add(documents=["hello world"], ids=["doc-a"])

        marker_path = os.path.join(palace_path, _EMBEDDER_MARKER_FILENAME)
        assert not os.path.exists(marker_path), (
            "marker file must NOT be written for legacy palaces"
        )

        result = coll.query(
            query_texts=["hi there"],
            n_results=1,
            include=["embeddings"],
        )
        assert len(result.ids[0]) == 1
        assert result.embeddings is not None
        # Legacy default = MiniLM-L6-v2 → 384-dim vectors.
        assert len(result.embeddings[0][0]) == 384
    finally:
        backend.close_palace(palace)
