"""Delegated children keep their own image-routing identity.

A pinned ``auxiliary.vision`` backend serves the MAIN conversation's vision
tool; it must not demote a vision-capable child's ``images:`` attachments to
describer text. ``delegation.image_input_mode`` pins children explicitly
(``auto`` — the default — lets the child's own model decide).
"""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

from agent import image_routing
from agent.delegation_context import delegated_child_context
from agent.image_routing import decide_image_input_mode
from hermes_constants import get_hermes_home


_TINY_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)

# The main conversation pins a describer (e.g. a text-only main on a vision-capable flash).
_AUX_PINNED = {
    "agent": {},
    "auxiliary": {
        "vision": {
            "provider": "zai",
            "model": "glm-5.3-flash",
            "base_url": "https://api.z.ai/api/coding/paas/v4",
        }
    },
}


def _force_vision(monkeypatch, verdict: bool) -> None:
    """Deterministic capability lookup — no catalog, no network."""
    monkeypatch.setattr(image_routing, "_lookup_supports_vision", lambda *a, **k: verdict)


def test_pinned_aux_keeps_main_conversation_text_mode(monkeypatch):
    _force_vision(monkeypatch, True)
    assert decide_image_input_mode("zai", "glm-5.3", _AUX_PINNED) == "text"


def test_pinned_aux_does_not_demote_vision_capable_child(monkeypatch):
    _force_vision(monkeypatch, True)
    with delegated_child_context("test-child"):
        assert decide_image_input_mode("zai", "glm-5.3-flash", _AUX_PINNED) == "native"


def test_child_on_text_only_model_stays_on_text_lane(monkeypatch):
    _force_vision(monkeypatch, False)
    with delegated_child_context("test-child"):
        assert decide_image_input_mode("zai", "glm-5.3", _AUX_PINNED) == "text"


def test_delegation_mode_text_pins_vision_capable_child(monkeypatch):
    _force_vision(monkeypatch, True)
    cfg = {**_AUX_PINNED, "delegation": {"image_input_mode": "text"}}
    with delegated_child_context("test-child"):
        assert decide_image_input_mode("zai", "glm-5.3-flash", cfg) == "text"


def test_delegation_mode_native_overrides_child_capability(monkeypatch):
    _force_vision(monkeypatch, False)
    cfg = {**_AUX_PINNED, "delegation": {"image_input_mode": "native"}}
    with delegated_child_context("test-child"):
        assert decide_image_input_mode("zai", "glm-5.3", cfg) == "native"


def test_delegation_mode_does_not_leak_into_main_conversation(monkeypatch):
    _force_vision(monkeypatch, True)
    cfg = {**_AUX_PINNED, "delegation": {"image_input_mode": "native"}}
    assert decide_image_input_mode("zai", "glm-5.3", cfg) == "text"


def test_multimodal_goal_embeds_pixels_inside_child_context(tmp_path, monkeypatch):
    """The spawn-site goal message honors the child scope: inside
    ``delegated_child_context`` a vision-capable child (lookup patched
    deterministically — no catalog, no network) gets image parts even when the
    persisted config pins ``auxiliary.vision``; outside the child scope the
    same pinned config keeps the hint lane (main-conversation contract
    unchanged — the spawn site is what wraps the call in the child context)."""
    home = Path(get_hermes_home())
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "auxiliary:\n"
        "  vision:\n"
        "    provider: zai\n"
        "    model: glm-5.3-flash\n",
        encoding="utf-8",
    )
    image = tmp_path / "img.png"
    image.write_bytes(_TINY_PNG)
    child = SimpleNamespace(provider="zai", model="child-model", requested_provider="")
    monkeypatch.setattr(image_routing, "_lookup_supports_vision", lambda *a, **k: True)

    from tools.delegate_tool_child_run import _build_child_goal_message

    with delegated_child_context("test-child"):
        parts = _build_child_goal_message("describe the image", [str(image)], child)
    assert isinstance(parts, list), f"expected multimodal parts, got {parts!r}"
    assert any(p.get("type") == "image_url" for p in parts)

    hints = _build_child_goal_message("describe the image", [str(image)], child)
    assert isinstance(hints, str) and "vision_analyze" in hints
