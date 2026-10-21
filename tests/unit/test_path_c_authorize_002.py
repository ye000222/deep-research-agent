"""Host publication exercised with synthetic history, no Docker or Provider."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.prospective_acquisition.path_c_attempts import (
    GenerationAttemptError,
    PathCAttemptAuthorization,
    digest,
)
from scripts.prospective_acquisition.path_c_authorize_002 import freeze_authorization, load_binding
from scripts.prospective_acquisition.path_c_generation import BATCH_ID, POPULATION_ID
from tests.unit.test_path_c_attempts import _history_fixture
from tests.unit.test_path_c_generation import _context


def test_success_immutable_no_dispatch(tmp_path: Path) -> None:
    _, batch, binding, raw = _history_fixture(tmp_path)
    with patch('httpx.Client.send', side_effect=AssertionError('outbound forbidden')) as send:
        target = freeze_authorization(batch, binding, expected_parent_hash=digest(raw))
    send.assert_not_called()
    auth = PathCAttemptAuthorization.model_validate_json(target.read_bytes())
    auth.validate_integrity()
    assert auth.attempt_number == 2
    assert auth.previous_state_sha256 == digest(raw)
    assert (batch / 'generation_state.json').read_bytes() == raw
    assert not (batch / 'attempt_002').exists()
    with pytest.raises(GenerationAttemptError):
        freeze_authorization(batch, binding, expected_parent_hash=digest(raw))
    assert PathCAttemptAuthorization.model_validate_json(target.read_bytes()) == auth


@pytest.mark.parametrize('field,value', [
    ('state', 'SUCCEEDED_AND_FROZEN'), ('physical_outbound_count', 0),
    ('population_id', 'wrong'), ('batch_id', 'wrong'),
])
def test_invalid_parent(tmp_path: Path, field: str, value: object) -> None:
    _, batch, binding, raw = _history_fixture(tmp_path)
    state = json.loads(raw)
    state[field] = value
    (batch / 'generation_state.json').write_text(json.dumps(state))
    with pytest.raises(GenerationAttemptError):
        freeze_authorization(batch, binding, expected_parent_hash=digest(raw))
    assert not (batch / 'authorizations').exists()


def test_hash_mismatch(tmp_path: Path) -> None:
    _, batch, binding, _ = _history_fixture(tmp_path)
    with pytest.raises(GenerationAttemptError):
        freeze_authorization(batch, binding, expected_parent_hash='0' * 64)
    assert not (batch / 'authorizations').exists()


@pytest.mark.parametrize('name', ['attempt_002', 'attempt_002.dispatch_reservation.json'])
def test_existing_evidence(tmp_path: Path, name: str) -> None:
    _, batch, binding, raw = _history_fixture(tmp_path)
    (batch / name).touch()
    with pytest.raises(GenerationAttemptError):
        freeze_authorization(batch, binding, expected_parent_hash=digest(raw))
    assert not (batch / 'authorizations').exists()


def test_publication_failure_no_partial_grant(tmp_path: Path) -> None:
    _, batch, binding, raw = _history_fixture(tmp_path)
    with (
        patch('os.link', side_effect=OSError('synthetic publication failure')),
        pytest.raises(OSError),
    ):
        freeze_authorization(batch, binding, expected_parent_hash=digest(raw))
    assert list((batch / 'authorizations').iterdir()) == []
    assert (batch / 'generation_state.json').read_bytes() == raw


def test_flush_failure_no_grant(tmp_path: Path) -> None:
    _, batch, binding, raw = _history_fixture(tmp_path)
    with patch('os.fsync', side_effect=OSError('synthetic flush failure')), pytest.raises(OSError):
        freeze_authorization(batch, binding, expected_parent_hash=digest(raw))
    assert list((batch / 'authorizations').iterdir()) == []
    assert (batch / 'generation_state.json').read_bytes() == raw


def test_frozen_binding_same_as_generation(tmp_path: Path) -> None:
    checkout, batch, binding, _ = _history_fixture(tmp_path)
    artifact_root = batch.parent.parent
    key = hashlib.sha256(json.dumps([POPULATION_ID, BATCH_ID], separators=(',', ':')).encode())
    folder = artifact_root / 'v2_1r_d4r_a6_closure' / 'execution_contexts'
    folder.mkdir(parents=True)
    context, _ = _context()
    target = folder / f'{key.hexdigest()}.json'
    target.write_text(json.dumps(context.safe_artifact()))
    assert load_binding(artifact_root, checkout) == binding
    artifact = json.loads(target.read_text())
    artifact['provider_binding']['profile_version'] = 24
    target.write_text(json.dumps(artifact))
    with pytest.raises(GenerationAttemptError, match='FROZEN_PROVIDER_BINDING_MISMATCH'):
        load_binding(artifact_root, checkout)
