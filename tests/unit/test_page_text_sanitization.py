"""Phase 14.3 stability guard: page-ingestion NUL byte sanitation.

These tests lock the storage-only boundary that strips un-storable 0x00 bytes
before page-derived text reaches PostgreSQL ``text`` columns.  The guard must
preserve every other byte (normal text is returned unchanged, byte-for-byte) and
must never disturb the content surrounding single or consecutive NULs.  It adds
no research capability and changes no decision, query, evidence, or closure
behavior.
"""

from __future__ import annotations

from app.infrastructure.db.research_tools import nul_safe_text


def test_normal_text_is_returned_byte_for_byte() -> None:
    value = "Industrial vision defect 检测 2024-2026, market size $1.2B.\n\ttabbed"
    result = nul_safe_text(value)
    assert result == value
    assert result.encode("utf-8") == value.encode("utf-8")


def test_single_nul_is_stripped_so_text_becomes_storable() -> None:
    result = nul_safe_text("before\x00after")
    assert result == "beforeafter"
    assert "\x00" not in result


def test_consecutive_nuls_do_not_affect_surrounding_content() -> None:
    result = nul_safe_text("head\x00\x00\x00tail")
    assert result == "headtail"
    assert "\x00" not in result


def test_leading_trailing_and_embedded_nuls_are_all_removed() -> None:
    assert nul_safe_text("\x00abc\x00def\x00") == "abcdef"


def test_empty_string_is_preserved() -> None:
    assert nul_safe_text("") == ""
