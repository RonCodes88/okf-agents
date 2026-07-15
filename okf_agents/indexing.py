"""Idempotent synchronization of OKF bundles into LangChain vector stores.

:func:`sync_bundle_to_vector_store` writes one document per concept using
stable, deterministic IDs so repeated runs add, update, or skip instead
of duplicating. The vector store owns its embedding implementation; this
module never touches embeddings directly.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from okf_agents.bundle import OKFBundle
from okf_agents.models import SyncResult
from okf_agents.retriever import concept_to_document

__all__ = ["stable_document_id", "sync_bundle_to_vector_store"]

CONTENT_HASH_KEY = "content_hash"

# Fixed namespace so stable IDs never collide with user UUIDs and stay
# identical across processes and machines for the same root + concept.
_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://okf.md/okf-agents/vector-ids")

_ERROR_MESSAGE_LIMIT = 200


def stable_document_id(bundle_root: str | Path, concept_id: str) -> str:
    """Return the deterministic vector-store ID for one concept.

    The ID is a UUIDv5 over the resolved bundle root and the concept ID,
    so the same concept always maps to the same document while distinct
    bundles never collide.
    """
    root = Path(bundle_root).resolve()
    return str(uuid.uuid5(_ID_NAMESPACE, f"{root}\x00{concept_id}"))


def _content_hash(document: Document) -> str:
    """Deterministic SHA-256 over page content and canonical metadata."""
    canonical = json.dumps(
        {"page_content": document.page_content, "metadata": document.metadata},
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _supports_stable_id_writes(store_type: type[VectorStore]) -> bool:
    """Whether the store overrides a write method that can accept ``ids``."""
    for name in ("add_documents", "add_texts"):
        method = getattr(store_type, name, None)
        if method is None or method is getattr(VectorStore, name, None):
            continue
        parameters = inspect.signature(method).parameters
        if "ids" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        ):
            return True
    return False


def _require_idempotent_store(vector_store: VectorStore) -> None:
    """Raise ``TypeError`` unless the store supports idempotent sync.

    The base ``VectorStore`` interface guarantees neither ID-based lookup
    nor stable-ID writes, so both capabilities are feature-detected
    before any mutation.
    """
    store_type = type(vector_store)
    if store_type.get_by_ids is VectorStore.get_by_ids:
        raise TypeError(
            f"{store_type.__name__} does not implement get_by_ids; idempotent "
            "synchronization requires ID-based document lookup"
        )
    if not _supports_stable_id_writes(store_type):
        raise TypeError(
            f"{store_type.__name__} does not accept stable document IDs; its "
            "add_documents/add_texts overrides take no `ids` argument"
        )


def _sanitize_error(concept_id: str, exc: Exception) -> str:
    """One bounded, concept-specific line without tracebacks or newlines."""
    message = str(exc).splitlines()[0] if str(exc) else ""
    line = f"{concept_id}: {type(exc).__name__}: {message}".rstrip(": ")
    return line[:_ERROR_MESSAGE_LIMIT]


def sync_bundle_to_vector_store(
    bundle: OKFBundle,
    vector_store: VectorStore,
    batch_size: int = 50,
    overwrite: bool = False,
) -> SyncResult:
    """Synchronize every concept in ``bundle`` into ``vector_store``.

    Each concept becomes one document with a stable ID derived from the
    resolved bundle root and concept ID, plus a deterministic
    ``content_hash`` metadata entry. Documents are classified as
    ``added`` (ID absent), ``skipped`` (present with an equal hash and
    ``overwrite=False``), ``updated`` (present but changed, or
    ``overwrite=True``), or ``failed`` (an attempted store operation
    raised; later batches still run).

    Raises:
        TypeError: If the store lacks ``get_by_ids`` or stable-ID writes.
        ValueError: If ``batch_size`` is less than 1.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    _require_idempotent_store(vector_store)

    prepared: list[tuple[str, str, Document]] = []
    for concept in bundle.all_concepts():
        document = concept_to_document(concept, bundle_root=bundle.root)
        document.metadata[CONTENT_HASH_KEY] = _content_hash(document)
        document.id = stable_document_id(bundle.root, concept.id)
        prepared.append((concept.id, document.id, document))

    added = updated = skipped = failed = 0
    errors: list[str] = []
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        try:
            existing = {
                found.id: found
                for found in vector_store.get_by_ids([doc_id for _, doc_id, _ in batch])
                if found.id is not None
            }
        except Exception as exc:
            failed += len(batch)
            errors.extend(_sanitize_error(concept_id, exc) for concept_id, _, _ in batch)
            continue

        writes: list[tuple[str, str, Document, bool]] = []
        batch_skipped = 0
        for concept_id, doc_id, document in batch:
            current = existing.get(doc_id)
            if current is None:
                writes.append((concept_id, doc_id, document, False))
            elif (
                not overwrite
                and current.metadata.get(CONTENT_HASH_KEY) == document.metadata[CONTENT_HASH_KEY]
            ):
                batch_skipped += 1
            else:
                writes.append((concept_id, doc_id, document, True))

        skipped += batch_skipped
        if not writes:
            continue
        try:
            vector_store.add_documents(
                [document for _, _, document, _ in writes],
                ids=[doc_id for _, doc_id, _, _ in writes],
            )
        except Exception as exc:
            failed += len(writes)
            errors.extend(_sanitize_error(concept_id, exc) for concept_id, _, _, _ in writes)
            continue
        added += sum(1 for _, _, _, is_update in writes if not is_update)
        updated += sum(1 for _, _, _, is_update in writes if is_update)

    return SyncResult(added=added, updated=updated, skipped=skipped, failed=failed, errors=errors)
