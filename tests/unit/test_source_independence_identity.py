"""Phase 15.1 §23 — Source Identity vs Source Role invariants (cases 1-9, 12, 15-17).

These tests lock the *identity* half of the phase's core lesson:

    "Do not mistake 100% webpage for only one independent source."

Independence must be keyed on the canonical publisher identity (registrable
domain / owner), never on the coarse ``source_role`` label.  Everything here is
question-, domain- and benchmark-independent (Generality Guard, spec §28): the
same normalisation rule is asserted across arbitrary synthetic hosts.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from uuid import UUID

from app.domain.evidence_alignment import EvidenceAlignment, EvidenceAlignmentStatus
from app.domain.gap_closure import project_gap_requirements
from app.domain.source_policy import normalize_source_url, source_owner_key

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_RUN_ID = UUID("00000000-0000-0000-0000-000000000123")


def _claim_requirement(*, required_sources: int, independent_sources: int):
    return project_gap_requirements(
        run_id=_RUN_ID,
        plan_version=1,
        state_version=4,
        coverage_map=[
            {
                "dimension_key": "qX",
                "requirement_statuses": [
                    {
                        "dimension_key": "qX:d1",
                        "coverage": 0.5,
                        "accepted_evidence": 3,
                        "independent_sources": independent_sources,
                        "required_sources": required_sources,
                    }
                ],
            }
        ],
        claim_states={"claim": {"dimension_key": "qX:d1", "unresolved": True}},
        now=_NOW,
    )[0]


# --- case 1: same URL -> same identity -------------------------------------
def test_identical_url_yields_identical_identity() -> None:
    url = "https://example.com/report/2024"
    assert source_owner_key(url) == source_owner_key(url)
    assert normalize_source_url(url) == normalize_source_url(url)


# --- case 2: www.example.com vs example.com -> same identity ---------------
def test_www_and_bare_domain_are_the_same_identity() -> None:
    assert source_owner_key("https://www.example.com/a") == source_owner_key(
        "https://example.com/b"
    )


# --- case 3: subdomains of one registered domain collapse ----------------
def test_subdomains_share_one_registered_domain_owner() -> None:
    owners = {
        source_owner_key(f"https://{sub}.google.com/{path}")
        for sub in ("research", "blog", "cloud")
        for path in ("x", "y")
    }
    assert owners == {"google.com"}


# --- case 4: different domains -> different identities ---------------------
def test_distinct_registrable_domains_are_distinct_identities() -> None:
    assert source_owner_key("https://www.nist.gov/x") != source_owner_key(
        "https://www.reuters.com/y"
    )
    assert source_owner_key("https://news.microsoft.com/z") != source_owner_key(
        "https://www.nature.org/w"
    )


# --- case 5: different organizations, same role -> independent -------------
def test_different_owners_are_independent_regardless_of_shared_role() -> None:
    # Same (webpage) role, different publishers -> distinct owner keys, so a
    # distinct-owner count keeps them independent.
    left = source_owner_key("https://nist.gov/statistics")
    right = source_owner_key("https://reuters.com/article")
    assert left != right
    assert len({left, right}) == 2


# --- case 6: same organization, different URLs -> not independent ----------
def test_same_owner_different_urls_is_one_source() -> None:
    urls = (
        "https://blogs.microsoft.com/2024/ai",
        "https://www.microsoft.com/en-us/research",
        "https://news.microsoft.com/press",
    )
    owners = {source_owner_key(url) for url in urls}
    assert len(owners) == 1


# --- case 7: role == webpage must not collapse identities ------------------
def test_webpage_role_does_not_collapse_independent_owners() -> None:
    # The corpus is "100% webpage" but spans multiple publishers; independence
    # is evaluated from the numeric distinct-owner count, so alignment closes
    # when the owner count meets the bar even though every role is identical.
    requirement = _claim_requirement(required_sources=2, independent_sources=2)
    alignment = EvidenceAlignment.evaluate(
        requirement,
        evidence_id=UUID(int=0x1),
        evidence_dimension_key=requirement.dimension_key,
        evidence_quality_passed=True,
        independent_source_count=2,  # two distinct owners, both role=webpage
        claim_verified=False,  # legacy hard-coded flag must no longer gate
        created_at=_NOW,
    )
    assert alignment.alignment_status is EvidenceAlignmentStatus.ALIGNED
    assert alignment.satisfies_requirement is True


# --- case 8: missing publisher -> fall back to domain ----------------------
def test_missing_publisher_falls_back_to_registrable_domain() -> None:
    # No publisher / organization metadata is available; owner key still
    # resolves to the registrable domain proxy rather than a role label.
    assert source_owner_key("https://research.insitech.co.uk/report") == "insitech.co.uk"


# --- case 9: missing all metadata -> conservative UNKNOWN ------------------
def test_missing_all_metadata_is_conservative_unknown() -> None:
    # Unattributable hosts collapse to a single conservative "unknown" owner so
    # they can never be counted as independent second sources.
    assert source_owner_key("not-a-url") == "unknown"
    assert source_owner_key("https://") == "unknown"


# --- case 12: duplicate evidence from one owner adds nothing ---------------
def test_duplicate_evidence_from_one_owner_does_not_inflate_count() -> None:
    evidence_urls = (
        "https://example.com/a",
        "https://www.example.com/b",
        "https://example.com/c?utm_source=x",
    )
    distinct_owners = {source_owner_key(url) for url in evidence_urls}
    assert len(distinct_owners) == 1


# --- case 15: genericity across arbitrary domains/questions ----------------
def test_identity_rule_is_generic_across_domains() -> None:
    # The same normalisation contract holds for arbitrary, non-benchmark hosts.
    pairs = (
        ("https://alpha.test/x", "https://www.alpha.test/y", True),
        ("https://beta.test/x", "https://gamma.test/y", False),
        ("https://one.co.jp/a", "https://www.one.co.jp/b", True),
    )
    for left, right, same in pairs:
        equal = source_owner_key(left) == source_owner_key(right)
        assert equal is same, (left, right)


# --- cases 16 & 17: no question-id / benchmark-specific branching ----------
def test_no_question_or_benchmark_specific_branching() -> None:
    source = inspect.getsource(source_owner_key)
    for token in ("q1", "q2", "q4", "question_id", "benchmark", "v1-dev"):
        assert token not in source, f"identity policy must not special-case {token!r}"
