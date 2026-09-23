"""Phase 15.1 §23 — Ground-Truth Audit classification (cases 18, 19, 20).

Covers the pure (database-free) helpers of the Task A analyzer so the branch
decision — *identity-collapse* vs *genuine shortage* — is locked by tests, not
just by one run of the audit.  The helpers must classify from numeric identity
counts and the persisted bar only, never from claim / question / domain text.
"""

from __future__ import annotations

from scripts.analyze_phase15_1_source_independence import (
    ALREADY_VERIFIED,
    GENUINE_SINGLE_SOURCE,
    IDENTITY_COLLAPSE,
    MISSING_SOURCE_METADATA,
    MIXED,
    UNKNOWN,
    classify_support,
    decide_branch,
    identity_collapse,
    is_unknown_owner,
)


# --- case 19: identity-collapse detection ----------------------------------
def test_identity_collapse_detected_when_domains_exceed_owners() -> None:
    # Several real domains squeezed down to a single owner == the identity bug.
    assert identity_collapse(observed_domains=2, observed_owners=1) is True
    assert identity_collapse(observed_domains=3, observed_owners=1) is True


def test_subdomain_folding_is_not_identity_collapse() -> None:
    # Many hosts under genuinely distinct publishers (owners track domains) is
    # correct normalisation, not a collapse.
    assert identity_collapse(observed_domains=1, observed_owners=1) is False
    assert identity_collapse(observed_domains=3, observed_owners=2) is False


def test_classify_flags_identity_collapse_below_bar() -> None:
    classification = classify_support(
        accepted=3,
        distinct_owners=1,
        distinct_domains=2,
        required=2,
        unknown_metadata=False,
    )
    assert classification == IDENTITY_COLLAPSE


# --- case 20: genuine-shortage detection -----------------------------------
def test_classify_flags_genuine_single_source() -> None:
    classification = classify_support(
        accepted=2,
        distinct_owners=1,
        distinct_domains=1,
        required=2,
        unknown_metadata=False,
    )
    assert classification == GENUINE_SINGLE_SOURCE


def test_classify_marks_already_verified_when_bar_met() -> None:
    classification = classify_support(
        accepted=2,
        distinct_owners=2,
        distinct_domains=2,
        required=2,
        unknown_metadata=False,
    )
    assert classification == ALREADY_VERIFIED


def test_classify_missing_metadata_is_conservative() -> None:
    classification = classify_support(
        accepted=2,
        distinct_owners=0,
        distinct_domains=0,
        required=2,
        unknown_metadata=True,
    )
    assert classification == MISSING_SOURCE_METADATA


def test_classify_unknown_without_accepted_evidence() -> None:
    classification = classify_support(
        accepted=0,
        distinct_owners=0,
        distinct_domains=0,
        required=2,
        unknown_metadata=False,
    )
    assert classification == UNKNOWN


def test_classify_mixed_when_multiple_owners_below_bar() -> None:
    # More than one owner but still short of the bar and no collapse: reported
    # MIXED so a partial shortage is never silently treated as verified.
    classification = classify_support(
        accepted=4,
        distinct_owners=2,
        distinct_domains=2,
        required=3,
        unknown_metadata=False,
    )
    assert classification == MIXED


# --- case 18: branch decision from aggregate shares ------------------------
def test_branch_b_when_genuine_shortage_dominates() -> None:
    decision = decide_branch({GENUINE_SINGLE_SOURCE: 44, IDENTITY_COLLAPSE: 0})
    assert decision["branch"] == "B"


def test_branch_a_when_identity_collapse_dominates() -> None:
    decision = decide_branch({IDENTITY_COLLAPSE: 30, GENUINE_SINGLE_SOURCE: 5})
    assert decision["branch"] == "A"


def test_branch_c_when_both_meaningful() -> None:
    decision = decide_branch({IDENTITY_COLLAPSE: 12, GENUINE_SINGLE_SOURCE: 30})
    assert decision["branch"] == "C"


def test_branch_none_when_nothing_unverified() -> None:
    decision = decide_branch({ALREADY_VERIFIED: 10})
    assert decision["branch"] == "NONE"


# --- unknown-owner policy --------------------------------------------------
def test_unknown_owner_placeholders_are_not_independent_sources() -> None:
    for value in ("", "   ", "unknown", "None", "NULL"):
        assert is_unknown_owner(value) is True
    assert is_unknown_owner("reuters.com") is False
