from uuid import uuid4

from app.domain.providers import CanonicalModelRequest, ContentPart


def test_provider_request_contract_never_contains_credentials() -> None:
    request = CanonicalModelRequest(
        task_kind="planning",
        role="planner",
        model="user-selected-model",
        instructions="Return a structured plan.",
        content_parts=(ContentPart(kind="text", value="Research the market"),),
        max_output_tokens=2048,
        context_manifest_id=uuid4(),
    )

    payload = request.model_dump(mode="json")

    assert "api_key" not in payload
    assert "credential" not in payload
    assert payload["model"] == "user-selected-model"
