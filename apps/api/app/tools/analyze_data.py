"""Declarative, evidence-bound analysis tool; never executes model-supplied code."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.controlled_tools import (
    AnalysisOperation,
    AnalyzeDataInput,
    AnalyzeDataResult,
)
from app.domain.identifiers import uuid7
from app.infrastructure.db.analysis_models import (
    AnalysisArtifactClaimRow,
    AnalysisArtifactRow,
    AnalysisInputRow,
)
from app.infrastructure.db.research_models import ResearchEvidenceRow, ResearchToolCallRow
from app.tools.errors import ToolExecutionError

_FORMULA_VERSION = "declarative-v1"


class AnalyzeDataTool:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def execute(self, request: AnalyzeDataInput) -> AnalyzeDataResult:
        evidence_ids = tuple(dict.fromkeys(item.evidence_id for item in request.data))
        duplicate_key = hashlib.sha256(
            json.dumps(request.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest()
        now = datetime.now(UTC)
        async with self._sessions() as session, session.begin():
            accepted_count = await session.scalar(
                select(func.count(ResearchEvidenceRow.id)).where(
                    ResearchEvidenceRow.run_id == request.run_id,
                    ResearchEvidenceRow.id.in_(evidence_ids),
                    ResearchEvidenceRow.accepted.is_(True),
                )
            )
            if int(accepted_count or 0) != len(evidence_ids):
                raise ToolExecutionError("ANALYSIS_EVIDENCE_INVALID", retryable=False)
            existing = await session.scalar(
                select(ResearchToolCallRow).where(
                    ResearchToolCallRow.run_id == request.run_id,
                    ResearchToolCallRow.tool_name == "analyze_data",
                    ResearchToolCallRow.duplicate_key == duplicate_key,
                )
            )
            if existing is not None:
                artifact = await session.scalar(
                    select(AnalysisArtifactRow).where(
                        AnalysisArtifactRow.tool_call_id == existing.id
                    )
                )
                if artifact is None:
                    raise ToolExecutionError("ANALYSIS_IDEMPOTENCY_CONFLICT", retryable=True)
                return _result_view(artifact, evidence_ids)
            call = ResearchToolCallRow(
                id=uuid7(),
                run_id=request.run_id,
                question_id=request.question_id,
                gap_id=request.target_gap_ids[0],
                action_id=request.action_id,
                tool_name="analyze_data",
                duplicate_key=duplicate_key,
                status="running",
                arguments=request.model_dump(mode="json"),
                result_refs={},
                started_at=now,
            )
            session.add(call)
            await session.flush()
            result, formula, warnings = _analyze(request)
            artifact = AnalysisArtifactRow(
                id=uuid7(),
                run_id=request.run_id,
                tool_call_id=call.id,
                question_id=request.question_id,
                operation=request.operation.value,
                parameters=request.parameters,
                formula=formula,
                formula_version=_FORMULA_VERSION,
                result=result,
                warnings=list(warnings),
                created_at=now,
            )
            session.add(artifact)
            claim_ids = tuple(
                dict.fromkeys(
                    item
                    for item in (
                        await session.scalars(
                            select(ResearchEvidenceRow.claim_id).where(
                                ResearchEvidenceRow.id.in_(evidence_ids),
                                ResearchEvidenceRow.claim_id.is_not(None),
                            )
                        )
                    ).all()
                    if item is not None
                )
            )
            for claim_id in claim_ids:
                session.add(
                    AnalysisArtifactClaimRow(
                        analysis_artifact_id=artifact.id,
                        claim_id=claim_id,
                        relation="derived_from",
                        confidence=1.0,
                        created_at=now,
                    )
                )
            for evidence_id in evidence_ids:
                session.add(
                    AnalysisInputRow(
                        analysis_artifact_id=artifact.id,
                        evidence_id=evidence_id,
                    )
                )
            call.status = "succeeded"
            call.result_refs = {"analysis_artifact_id": str(artifact.id)}
            call.finished_at = now
            return _result_view(artifact, evidence_ids)


def _analyze(request: AnalyzeDataInput) -> tuple[dict[str, object], str, tuple[str, ...]]:
    values = [item.value for item in request.data]
    operation = request.operation
    if operation is AnalysisOperation.AGGREGATE:
        method = str(request.parameters.get("method", "sum"))
        functions: dict[str, Callable[[list[float]], float]] = {
            "sum": lambda values: float(sum(values)),
            "mean": lambda values: float(statistics.fmean(values)),
            "min": lambda values: float(min(values)),
            "max": lambda values: float(max(values)),
        }
        if method not in functions:
            raise ToolExecutionError("ANALYSIS_PARAMETER_INVALID", retryable=False)
        value = float(functions[method](values))
        return {"method": method, "value": value, "unit": request.data[0].unit}, method, ()
    if operation is AnalysisOperation.GROWTH_RATE:
        _require_points(values, 2)
        if values[0] == 0:
            raise ToolExecutionError("ANALYSIS_DIVISION_BY_ZERO", retryable=False)
        growth = (values[-1] - values[0]) / abs(values[0])
        return {"growth_rate": growth}, "(last - first) / abs(first)", ()
    if operation is AnalysisOperation.CAGR:
        _require_points(values, 2)
        periods = float(request.parameters.get("periods", len(values) - 1))
        if values[0] <= 0 or values[-1] < 0 or periods <= 0:
            raise ToolExecutionError("ANALYSIS_PARAMETER_INVALID", retryable=False)
        cagr = (values[-1] / values[0]) ** (1.0 / periods) - 1.0
        return {"cagr": cagr, "periods": periods}, "(last / first) ** (1 / periods) - 1", ()
    if operation is AnalysisOperation.RATIO:
        _require_exact_points(values, 2)
        if values[1] == 0:
            raise ToolExecutionError("ANALYSIS_DIVISION_BY_ZERO", retryable=False)
        return {"ratio": values[0] / values[1]}, "first / second", ()
    if operation is AnalysisOperation.RANK:
        descending = bool(request.parameters.get("descending", True))
        ranked = sorted(request.data, key=lambda item: item.value, reverse=descending)
        return (
            {
                "ranking": [
                    {"rank": index, "label": item.label, "value": item.value, "unit": item.unit}
                    for index, item in enumerate(ranked, start=1)
                ]
            },
            "stable sort(value)",
            (),
        )
    if operation is AnalysisOperation.COMPARE:
        _require_exact_points(values, 2)
        delta = values[1] - values[0]
        percent = None if values[0] == 0 else delta / abs(values[0])
        warnings = ("baseline_is_zero",) if percent is None else ()
        return {"delta": delta, "percent_change": percent}, "second - first", warnings
    if operation is AnalysisOperation.DESCRIPTIVE_STATS:
        mean = statistics.fmean(values)
        variance = statistics.fmean([(value - mean) ** 2 for value in values])
        return (
            {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": mean,
                "median": statistics.median(values),
                "stddev_population": math.sqrt(variance),
                "unit": request.data[0].unit,
            },
            "population descriptive statistics",
            (),
        )
    raise ToolExecutionError("ANALYSIS_OPERATION_UNSUPPORTED", retryable=False)


def _require_points(values: list[float], minimum: int) -> None:
    if len(values) < minimum:
        raise ToolExecutionError("ANALYSIS_DATA_INSUFFICIENT", retryable=False)


def _require_exact_points(values: list[float], expected: int) -> None:
    if len(values) != expected:
        raise ToolExecutionError("ANALYSIS_DATA_CARDINALITY_INVALID", retryable=False)


def _result_view(
    artifact: AnalysisArtifactRow, evidence_ids: tuple[UUID, ...]
) -> AnalyzeDataResult:
    return AnalyzeDataResult(
        artifact_id=artifact.id,
        call_id=artifact.tool_call_id,
        status="success",
        operation=AnalysisOperation(artifact.operation),
        result=artifact.result,
        formula=artifact.formula,
        formula_version=artifact.formula_version,
        input_evidence_ids=evidence_ids,
        warnings=tuple(artifact.warnings),
    )
