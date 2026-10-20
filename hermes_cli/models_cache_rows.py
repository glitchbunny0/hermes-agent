"""Validity rules for rows of the provider-model disk cache (``provider_models_cache.json``).

Split out of ``hermes_cli.models`` so the facade does not grow; the row layout and the fingerprint
those rows are keyed on are documented there (``_credential_fingerprint``).
"""

from __future__ import annotations

from typing import Any, TypeGuard


def cache_entry_valid(
    entry: Any, fp: str, *, allow_empty: bool = False) -> TypeGuard[dict[str, Any]]:
    """Well-formed cache row for fingerprint *fp*. Requires a numeric ``at`` so corrupt disk state
    degrades to a cache miss instead of raising; empty model lists are valid only when the caller
    opts into an authoritative empty catalog."""
    return (
        isinstance(entry, dict)
        and entry.get("fp") == fp
        and isinstance(entry.get("models"), list)
        and (allow_empty or bool(entry["models"]))
        and isinstance(entry.get("at"), (int, float))
        and not isinstance(entry.get("at"), bool))


def same_credential_identity(entry: Any, fp: str) -> TypeGuard[dict[str, Any]]:
    """*entry* is a real (non-empty, non-fallback) catalog whose fingerprint differs from *fp* only
    in its credential-file mtimes (the ``-<files>`` half of ``_credential_fingerprint``).

    Any provider's credential write (a rate-limit flip, an OAuth refresh) moves those mtimes. The GUI
    read path serves such a row while a background refresh re-stamps it, because the curated list
    it would otherwise fall back to can be shorter than the account's real catalog. A row that
    differs in the identity half (env values, endpoint) or lacks the split format never matches."""
    stored = entry.get("fp") if isinstance(entry, dict) else None
    return (isinstance(stored, str) and "-" in stored and "-" in fp
            and stored.partition("-")[0] == fp.partition("-")[0]
            and not entry.get("fallback")
            and cache_entry_valid(entry, stored))
