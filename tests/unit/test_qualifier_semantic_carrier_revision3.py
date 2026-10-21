# ruff: noqa: RUF001
from __future__ import annotations

from app.services.atomic_proposition_validation import (
    QualifierCheckStatus,
    _proposition_semantic_view,
    inspect_qualifier_fields,
)


def _checks(claim: str, *propositions: dict[str, object]) -> dict[str, object]:
    return {item.category: item for item in inspect_qualifier_fields(claim, list(propositions))}


def test_method_semantics_in_comparison_baseline_carrier_are_preserved() -> None:
    claim = "相比传统基于灰度阈值、边缘检测等方法，实例分割具有更高精度。"
    result = _checks(
        claim,
        {
            "local_ref": "p0",
            "proposition_text": "实例分割比传统基于灰度阈值、边缘检测等方法精度更高。",
            "subject": "实例分割",
            "predicate": "精度更高",
            "source_span": claim,
            "typed_value": {
                "kind": "comparison",
                "comparison_target": "传统基于灰度阈值、边缘检测等方法",
            },
            "material_qualifiers": [
                {
                    "category": "comparison_target",
                    "key": "baseline_methods",
                    "value": "传统基于灰度阈值、边缘检测等方法",
                    "source_span": "传统基于灰度阈值、边缘检测等方法",
                }
            ],
        },
    )["method"]
    assert result.status is QualifierCheckStatus.PRESERVED
    evidence = [item for item in result.preservation_evidence if item.semantic_compatible]
    assert evidence
    assert any("comparison_target" in path for item in evidence for path in item.carrier_paths)


def test_structured_other_configuration_carrier_is_role_and_value_checked() -> None:
    claim = "该文将视频跟踪模式用于缺陷分割。"
    result = _checks(
        claim,
        {
            "local_ref": "p0",
            "proposition_text": claim,
            "subject": "该文",
            "predicate": "将模式用于缺陷分割",
            "source_span": claim,
            "material_qualifiers": [
                {
                    "category": "other",
                    "key": "使用模式",
                    "value": "视频跟踪模式",
                    "source_span": "视频跟踪模式",
                }
            ],
        },
    )["configuration"]
    assert result.status is QualifierCheckStatus.PRESERVED
    assert any(
        "material_qualifiers[0].value" in path
        for item in result.preservation_evidence
        for path in item.carrier_paths
    )


def test_referential_method_subject_does_not_create_an_obligation() -> None:
    result = _checks(
        "该方法先拟合正常特征，然后计算变换。",
        {
            "local_ref": "p0",
            "proposition_text": "该方法先拟合正常特征。",
            "subject": "该方法",
            "predicate": "先拟合正常特征",
            "source_span": "该方法先拟合正常特征",
        },
        {
            "local_ref": "p1",
            "proposition_text": "该方法计算变换。",
            "subject": "该方法",
            "predicate": "计算变换",
            "source_span": "然后计算变换",
        },
    )["method"]
    assert result.status is QualifierCheckStatus.NOT_APPLICABLE
    assert result.obligation_count == 0


def test_asserted_method_subject_keeps_reviewed_attachment_ambiguity_visible() -> None:
    claim = (
        "传统方法受限于手工特征或领域专用专家模型，泛化能力不足，"
        "而端到端MLLM方案在泛化性和可解释性上更优且数据需求更低。"
    )
    propositions = [
        {
            "local_ref": f"p{index}",
            "subject": subject,
            "predicate": predicate,
            "proposition_text": text,
            "typed_value": typed_value,
            "material_qualifiers": [],
        }
        for index, (subject, predicate, text, typed_value) in enumerate(
            [
                (
                    "传统方法",
                    "受限于",
                    "传统方法受限于手工特征或领域专用专家模型。",
                    {"kind": "set", "members": ["手工特征", "领域专用专家模型"]},
                ),
                (
                    "传统方法",
                    "泛化能力不足",
                    "传统方法泛化能力不足。",
                    {"kind": "text", "text": "泛化能力不足"},
                ),
                (
                    "端到端MLLM方案",
                    "在泛化性上更优",
                    "端到端MLLM方案在泛化性上更优。",
                    {
                        "kind": "comparison",
                        "comparison_target": "传统方法",
                        "comparison_metric": "泛化性",
                    },
                ),
                (
                    "端到端MLLM方案",
                    "在可解释性上更优",
                    "端到端MLLM方案在可解释性上更优。",
                    {
                        "kind": "comparison",
                        "comparison_target": "传统方法",
                        "comparison_metric": "可解释性",
                    },
                ),
                (
                    "端到端MLLM方案",
                    "数据需求更低",
                    "端到端MLLM方案数据需求更低。",
                    {
                        "kind": "comparison",
                        "comparison_target": "传统方法",
                        "comparison_metric": "数据需求",
                    },
                ),
            ]
        )
    ]
    result = _checks(claim, *propositions)["method"]
    assert result.status is QualifierCheckStatus.AMBIGUOUS
    assert result.obligation_count == 1


def test_comparison_target_with_wrong_method_value_does_not_satisfy_method() -> None:
    claim = "系统基于方法 Alpha 执行检测。"
    result = _checks(
        claim,
        {
            "local_ref": "p0",
            "proposition_text": "系统执行检测。",
            "subject": "系统",
            "predicate": "执行检测",
            "source_span": claim,
            "typed_value": {"kind": "comparison", "comparison_target": "method Beta"},
        },
    )["method"]
    assert result.status is QualifierCheckStatus.POTENTIAL_LOSS


def test_baseline_method_on_another_proposition_cannot_be_borrowed() -> None:
    claim = "Device A uses method Alpha; Device B uses method Beta."
    result = _checks(
        claim,
        {
            "local_ref": "p1",
            "proposition_text": "Device B uses method Beta.",
            "subject": "Device B",
            "predicate": "uses",
            "source_span": "Device B uses method Beta",
            "material_qualifiers": [
                {
                    "category": "method",
                    "key": "method",
                    "value": "method Beta",
                    "source_span": "uses method Beta",
                }
            ],
        },
    )["method"]
    assert result.status in {
        QualifierCheckStatus.POTENTIAL_LOSS,
        QualifierCheckStatus.AMBIGUOUS,
    }
    first_obligation = result.preservation_evidence[0]
    assert first_obligation.proposition_index is None


def test_other_value_with_unrelated_role_is_not_a_configuration_carrier() -> None:
    claim = "系统在 Alpha 模式下运行。"
    result = _checks(
        claim,
        {
            "local_ref": "p0",
            "proposition_text": "系统运行。",
            "subject": "系统",
            "predicate": "运行",
            "source_span": claim,
            "material_qualifiers": [
                {
                    "category": "other",
                    "key": "display label",
                    "value": "Alpha 模式",
                    "source_span": "Alpha 模式",
                }
            ],
        },
    )["configuration"]
    assert result.status is QualifierCheckStatus.POTENTIAL_LOSS


def test_configuration_role_with_wrong_attachment_cannot_satisfy_other_mode() -> None:
    claim = "Under mode Alpha, Device A runs; Device B idles under mode Beta."
    result = _checks(
        claim,
        {
            "local_ref": "p0",
            "proposition_text": "Device A runs.",
            "subject": "Device A",
            "predicate": "runs",
            "source_span": "Under mode Alpha, Device A runs",
        },
        {
            "local_ref": "p1",
            "proposition_text": "Device B idles under mode Beta.",
            "subject": "Device B",
            "predicate": "idles",
            "source_span": "Device B idles under mode Beta",
            "material_qualifiers": [
                {
                    "category": "other",
                    "key": "configuration",
                    "value": "mode Beta",
                    "source_span": "mode Beta",
                }
            ],
        },
    )["configuration"]
    assert result.status is not QualifierCheckStatus.PRESERVED


def test_method_relation_still_blocks_when_its_method_object_was_dropped() -> None:
    claim = "The system uses method Alpha for inspection."
    result = _checks(
        claim,
        {
            "local_ref": "p0",
            "proposition_text": "The system uses a method for inspection.",
            "subject": "system",
            "predicate": "uses a method for inspection",
            "source_span": claim,
        },
    )["method"]
    assert result.status is QualifierCheckStatus.POTENTIAL_LOSS


def test_carrier_records_role_path_and_proposition_provenance() -> None:
    claim = "System runs using method Alpha."
    result = _checks(
        claim,
        {
            "local_ref": "p7",
            "proposition_text": claim,
            "subject": "System",
            "predicate": "runs",
            "source_span": claim,
            "typed_value": {"kind": "text", "text": "using method Alpha"},
        },
    )["method"]
    matching = [
        carrier for evidence in result.preservation_evidence for carrier in evidence.carrier_paths
    ]
    assert result.status is QualifierCheckStatus.PRESERVED
    assert matching
    view = _proposition_semantic_view(
        0,
        {
            "local_ref": "p7",
            "typed_value": {"kind": "text", "text": "using method Alpha"},
        },
    )
    assert view.proposition_ref == "p7"
    assert any(carrier.field_path == "typed_value.text" for carrier in view.carriers)
