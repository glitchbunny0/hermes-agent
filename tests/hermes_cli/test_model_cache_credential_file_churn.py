"""The GUI picker must not fall back to the curated list because a credential FILE was rewritten.

``auth.json`` is rewritten by every pool status flip, rate-limit mark and OAuth refresh of ANY
provider. The cache fingerprint carries that file's mtime, so each write used to make every
provider's row look cold; the non-blocking picker read then returned ``[]`` and the caller showed
the static curated list, which can be shorter than the account's real catalog, until the background
refresh re-stamped the row. Only a change in what the credentials ARE (env values, endpoint) may
drop the row.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

import hermes_cli.models as mod

LIVE = ["claude-opus-5", "claude-sonnet-5-5", "claude-haiku-5-5"]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _write_auth(home, mtime_ns: int) -> None:
    path = home / "auth.json"
    path.write_text(json.dumps({"version": 1, "updated_at": str(mtime_ns)}), encoding="utf-8")
    os.utime(path, ns=(mtime_ns, mtime_ns))


def _seed(models, **extra) -> None:
    fp = mod._credential_fingerprint("anthropic")
    mod._store_cache_entry("anthropic", {**mod._cache_entry(fp, models), **extra})


def test_credential_file_write_keeps_the_real_catalog_on_the_gui_read(home):
    _write_auth(home, 1_000_000_000)
    _seed(LIVE)

    _write_auth(home, 2_000_000_000)  # unrelated pool write: mtime moves, credentials do not
    with patch.object(mod, "_spawn_swr_refresh") as spawn:
        assert mod.cached_provider_model_ids("anthropic", non_blocking=True) == LIVE
    spawn.assert_called_once_with("anthropic")


def test_changed_credentials_still_drop_the_row(home, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-a")
    _write_auth(home, 1_000_000_000)
    _seed(LIVE)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-b")
    with patch.object(mod, "_spawn_swr_refresh") as spawn:
        assert mod.cached_provider_model_ids("anthropic", non_blocking=True) == []
    spawn.assert_called_once_with("anthropic")


def test_curated_fallback_row_is_not_promoted_by_a_file_write(home):
    _write_auth(home, 1_000_000_000)
    _seed(LIVE, fallback=True)

    _write_auth(home, 2_000_000_000)
    with patch.object(mod, "_spawn_swr_refresh"):
        assert mod.cached_provider_model_ids("anthropic", non_blocking=True) == []


def test_blocking_read_still_refetches_after_a_file_write(home):
    _write_auth(home, 1_000_000_000)
    _seed(LIVE)

    _write_auth(home, 2_000_000_000)
    with patch.object(mod, "provider_model_ids", return_value=["fresh"]) as live:
        assert mod.cached_provider_model_ids("anthropic") == ["fresh"]
    live.assert_called_once()


def test_fingerprint_halves_track_identity_and_files_separately(home, monkeypatch):
    _write_auth(home, 1_000_000_000)
    base_identity, base_files = mod._credential_fingerprint("anthropic").split("-")

    _write_auth(home, 2_000_000_000)
    identity, files = mod._credential_fingerprint("anthropic").split("-")
    assert (identity, files != base_files) == (base_identity, True)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "key-a")
    assert mod._credential_fingerprint("anthropic").split("-")[0] != base_identity
