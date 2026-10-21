"""Reference-space reconciliation for offline Phase C replay artifacts.

This module only normalizes persisted identifiers and eligibility lookups. It
does not infer or alter proposition/composition semantics.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def _eligibility_entries(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    if isinstance(raw, list):
        result: dict[str, str] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            key = item.get("ref", item.get("proposition_ref", item.get("local_ref")))
            value = item.get("eligibility", item.get("status"))
            if key is not None and value is not None:
                result[str(key)] = str(value)
        return result
    return {}


def reconcile_references(
    raw_propositions: list[dict[str, Any]],
    model_local_references: Any,
    raw_eligibility: Any,
) -> dict[str, Any]:
    """Return canonical proposition identities, eligibility aliases, and issues.

    The persisted ``model_local_references`` contract in Final-Unseen artifacts
    is local-ref -> runtime-id. Direct-reference cohorts omit that mapping.
    """
    issues: list[dict[str, str]] = []
    local_to_runtime: dict[str, str] = {}
    if model_local_references is not None:
        if not isinstance(model_local_references, dict):
            issues.append({"code": "LOCAL_REF_MAPPING_MISSING", "detail": "mapping_not_object"})
        else:
            for local, runtime in model_local_references.items():
                if (
                    not isinstance(local, str)
                    or not isinstance(runtime, str)
                    or not local
                    or not runtime
                ):
                    issues.append(
                        {"code": "LOCAL_REF_MAPPING_MISSING", "detail": "invalid_mapping_pair"}
                    )
                    continue
                if local in local_to_runtime:
                    issues.append({"code": "DUPLICATE_LOCAL_REF", "detail": local})
                local_to_runtime[local] = runtime

    runtime_to_locals: dict[str, list[str]] = {}
    for local, runtime in local_to_runtime.items():
        runtime_to_locals.setdefault(runtime, []).append(local)
    for runtime, locals_ in runtime_to_locals.items():
        if len(locals_) > 1:
            issues.append({"code": "DUPLICATE_RUNTIME_ID", "detail": runtime})

    eligibility = _eligibility_entries(raw_eligibility)
    normalized: list[dict[str, Any]] = []
    used_eligibility_keys: set[str] = set()
    used_local_refs: set[str] = set()
    used_runtime_ids: set[str] = set()
    for index, raw in enumerate(raw_propositions):
        raw_ref_value = raw.get("proposition_id", raw.get("runtime_proposition_id"))
        raw_local_value = raw.get("local_ref", raw.get("model_local_ref"))
        raw_ref = str(raw_ref_value) if raw_ref_value is not None else None
        raw_local = str(raw_local_value) if raw_local_value is not None else None

        local_ref: str | None = raw_local
        runtime_id: str | None = raw_ref
        identity_kind = "RUNTIME_PROPOSITION_ID" if raw_ref else "MODEL_LOCAL_REF"

        if local_to_runtime:
            if raw_local and raw_local in local_to_runtime:
                local_ref = raw_local
                mapped_runtime = local_to_runtime[raw_local]
                if raw_ref and raw_ref != mapped_runtime:
                    issues.append(
                        {
                            "code": "RUNTIME_ID_MAPPING_MISSING",
                            "detail": f"{raw_local}:{raw_ref}!={mapped_runtime}",
                        }
                    )
                runtime_id = raw_ref or mapped_runtime
                if runtime_id != mapped_runtime:
                    issues.append({"code": "RUNTIME_ID_MAPPING_MISSING", "detail": raw_local})
                identity_kind = "RUNTIME_PROPOSITION_ID"
            elif raw_ref and raw_ref in runtime_to_locals:
                runtime_id = raw_ref
                aliases = runtime_to_locals[raw_ref]
                if len(aliases) == 1:
                    local_ref = aliases[0]
                identity_kind = "RUNTIME_PROPOSITION_ID"
            elif raw_ref and raw_ref in local_to_runtime:
                local_ref = raw_ref
                runtime_id = local_to_runtime[raw_ref]
                identity_kind = "RUNTIME_PROPOSITION_ID"
            else:
                issues.append(
                    {"code": "LOCAL_REF_MAPPING_MISSING", "detail": raw_ref or f"index:{index}"}
                )
                runtime_id = raw_ref
        else:
            local_ref = raw_local or raw_ref
            runtime_id = raw_ref or raw_local
            identity_kind = (
                "MODEL_LOCAL_REF"
                if runtime_id and runtime_id.startswith("p")
                else "RUNTIME_PROPOSITION_ID"
            )

        if not runtime_id:
            issues.append({"code": "RUNTIME_ID_MAPPING_MISSING", "detail": f"index:{index}"})
            runtime_id = f"UNRESOLVED:{index}"
        if local_ref:
            if local_ref in used_local_refs:
                issues.append({"code": "DUPLICATE_LOCAL_REF", "detail": local_ref})
            used_local_refs.add(local_ref)
        if runtime_id in used_runtime_ids:
            issues.append({"code": "DUPLICATE_RUNTIME_ID", "detail": runtime_id})
        used_runtime_ids.add(runtime_id)

        lookup_keys = [key for key in (local_ref, runtime_id, raw_ref, raw_local) if key]
        matched_key = next((key for key in lookup_keys if key in eligibility), None)
        if matched_key is None:
            value = "UNKNOWN"
            issues.append({"code": "PROPOSITION_WITHOUT_ELIGIBILITY_MAPPING", "detail": runtime_id})
        else:
            value = eligibility[matched_key]
            used_eligibility_keys.add(matched_key)
        normalized.append(
            {
                **raw,
                "local_ref": runtime_id,
                "runtime_proposition_id": runtime_id,
                "model_local_ref": local_ref,
                "reference_identity_kind": identity_kind,
                "eligibility": value,
                "eligibility_source_key": matched_key,
                "eligibility_reason": None
                if matched_key
                else "SOURCE_ARTIFACT_HAS_NO_RESOLVABLE_RESULT",
            }
        )

    for key in eligibility.keys() - used_eligibility_keys:
        issues.append({"code": "ORPHAN_ELIGIBILITY_ENTRY", "detail": key})

    return {
        "propositions": normalized,
        "eligibility_by_ref": {
            item["runtime_proposition_id"]: item["eligibility"] for item in normalized
        },
        "local_to_runtime": local_to_runtime,
        "runtime_to_local": {
            runtime: locals_[0]
            for runtime, locals_ in runtime_to_locals.items()
            if len(locals_) == 1
        },
        "issues": issues,
        "counts": {
            "propositions": len(normalized),
            "eligibility_source_entries": len(eligibility),
            "local_ref_mappings": len(local_to_runtime),
            "reconciled_propositions": sum(
                item["eligibility_source_key"] is not None for item in normalized
            ),
            "unresolved_eligibility": sum(
                item["eligibility_source_key"] is None for item in normalized
            ),
            "orphan_eligibility_entries": sum(
                issue["code"] == "ORPHAN_ELIGIBILITY_ENTRY" for issue in issues
            ),
            "duplicate_mappings": sum(
                issue["code"] in {"DUPLICATE_LOCAL_REF", "DUPLICATE_RUNTIME_ID"} for issue in issues
            ),
        },
        "issue_counts": dict(sorted(Counter(issue["code"] for issue in issues).items())),
    }


def reconcile_member_ref(ref: str, local_to_runtime: dict[str, str]) -> str:
    """Resolve a composition member alias without interpreting its operator."""
    return local_to_runtime.get(ref, ref)
