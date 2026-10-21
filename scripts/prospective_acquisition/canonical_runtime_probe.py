import sys
_stage = "PROBE_PROCESS_STARTED"
_modules = []
_identity = {"status": "NOT_EVALUATED", "files": []}
_db = {"status": "NOT_EVALUATED"}
_provider = {"status": "NOT_REACHED", "profiles": []}
_safe_codes = {
    "PROBE_PROCESS_STARTED": "PYTHON_PROBE_STARTUP_FAILED",
    "PYTHON_ENV_READY": "PYTHON_PROBE_STARTUP_FAILED",
    "CURRENT_CHECKOUT_IDENTITY": "CURRENT_CHECKOUT_IDENTITY_FAILED",
    "IMPORT_PROBE": "RUNTIME_IMPORT_FAILED",
    "DB_CONNECTION": "DATABASE_READ_PROBE_FAILED",
    "DB_READONLY": "DATABASE_READ_PROBE_FAILED",
    "DB_QUERY": "DATABASE_READ_PROBE_FAILED",
    "PROVIDER_PROFILE_RESOLUTION": "PROVIDER_PROFILE_RESOLUTION_FAILED",
    "PROVIDER_SAFE_FINGERPRINT": "PROVIDER_SAFE_FINGERPRINT_FAILED",
    "FROZEN_CONTEXT": "FROZEN_EXECUTION_CONTEXT_INVALID",
}
def _marker(name, **safe_fields):
    global _stage
    _stage = name
    item = {"stage": name}
    item.update(safe_fields)
    sys.stderr.write("D4RA6_STAGE " + __import__("json").dumps(item, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stderr.flush()
def _envelope(status, **fields):
    return {"attestation_schema": "d4ra6_runtime_probe.v2", "status": status, **fields}
def _validate_frozen_context(context, expected_context_id, expected_population_id, expected_batch_id):
    import uuid
    required_binding = ("profile_id","adapter","model","endpoint_family","profile_version","credential_version_id","credential_version","credential_present","fingerprint_contract","provider_safe_config_fingerprint")
    binding = context.get("provider_binding") if isinstance(context,dict) else None
    if (
        not isinstance(context,dict)
        or context.get("contract_version") != "prospective-acquisition-execution-context.v1"
        or context.get("execution_context_id") != expected_context_id
        or context.get("population_id") != expected_population_id
        or context.get("batch_id") != expected_batch_id
        or context.get("owner_scope_source") != "AUTHENTICATED_CLIENT_SESSION"
        or context.get("provider_resolution_policy") != "FROZEN_DEFAULT_PROFILE_V1"
        or not isinstance(context.get("context_hash"),str)
        or len(context["context_hash"]) != 64
        or any(ch not in "0123456789abcdef" for ch in context["context_hash"])
        or not isinstance(context.get("owner_scope_fingerprint"),str)
        or len(context["owner_scope_fingerprint"]) != 64
        or any(ch not in "0123456789abcdef" for ch in context["owner_scope_fingerprint"])
        or not isinstance(binding,dict)
        or any(field not in binding for field in required_binding)
        or set(binding) != set(required_binding)
        or binding.get("fingerprint_contract") != "provider-safe-config-fingerprint.v1"
        or not isinstance(binding.get("provider_safe_config_fingerprint"),str)
        or len(binding["provider_safe_config_fingerprint"]) != 64
        or any(ch not in "0123456789abcdef" for ch in binding["provider_safe_config_fingerprint"])
        or binding.get("credential_present") is not True
        or not isinstance(binding.get("profile_version"),int)
        or isinstance(binding.get("profile_version"),bool)
        or not isinstance(binding.get("credential_version"),int)
        or isinstance(binding.get("credential_version"),bool)
        or not isinstance(context.get("protocol_version"),str)
        or not context["protocol_version"]
        or not isinstance(context.get("created_at"),str)
        or not context["created_at"]
    ):
        raise ValueError("frozen_context_contract_invalid")
    uuid.UUID(binding["profile_id"])
    uuid.UUID(binding["credential_version_id"])
    return binding
def _compute_frozen_context_hash(context, owner_hash):
    import hashlib, json
    payload = {
        "execution_context_id":context["execution_context_id"],
        "population_id":context["population_id"],
        "batch_id":context["batch_id"],
        "owner_hash":owner_hash,
        "owner_scope_source":context["owner_scope_source"],
        "provider_resolution_policy":context["provider_resolution_policy"],
        "provider_binding":context["provider_binding"],
        "protocol_version":context["protocol_version"],
        "created_at":context["created_at"],
    }
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def _provider_binding_matches(frozen, current):
    required = ("profile_id","adapter","model","endpoint_family","profile_version","credential_version_id","credential_version","credential_present","fingerprint_contract","provider_safe_config_fingerprint")
    return isinstance(frozen,dict) and isinstance(current,dict) and all(field in frozen and field in current and frozen[field] == current[field] for field in required) and set(frozen) == set(required) and set(current) == set(required)
def _safe_context_projection(context, artifact_path, binding_match):
    return {
        "status":"PASS",
        "execution_context_id":context["execution_context_id"],
        "artifact_contract_version":context["contract_version"],
        "population_id":context["population_id"],
        "batch_id":context["batch_id"],
        "owner_scope_source":context["owner_scope_source"],
        "owner_scope_fingerprint":context["owner_scope_fingerprint"],
        "context_hash":context["context_hash"],
        "artifact_path":artifact_path,
        "provider_binding":context["provider_binding"],
        "provider_binding_match":binding_match,
    }
def _safe_fail(exc):
    import json
    code = getattr(exc, "safe_code", None) or _safe_codes.get(_stage, "PROBE_FAILURE")
    payload = _envelope(
        "FAILED", failed_stage=_stage, exception_type=type(exc).__name__, safe_message=code,
        imports=_modules, current_checkout_identity=_identity, database=_db, provider=_provider,
        provider_generation_calls=0, provider_http_count=0, semantic_verifier_calls=0,
        db_writes=0, holdout_access=0, schema_runtime=globals().get('_schema_runtime'),
    )
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()
class _ProbeFailure(Exception):
    def __init__(self, safe_code):
        super().__init__(safe_code)
        self.safe_code = safe_code
try:
    _marker("PROBE_PROCESS_STARTED")
    import asyncio, hashlib, importlib, json, os, pathlib, platform, base64
    sys.path.insert(0, "/app/apps/api")
    sys.path.insert(0, "/opt")
    sys.path.insert(0, "/opt/scripts")
    _expected = json.loads(base64.b64decode(os.environ["OPERATOR_EXPECTED_HASHES_B64"]).decode("utf-8"))
    _cwd = str(pathlib.Path.cwd().resolve())
    _selected_path = [p for p in sys.path if p in ("/app/apps/api", "/opt", "/opt/scripts")]
    _marker("PYTHON_ENV_READY", python_version=platform.python_version(), executable=sys.executable, platform=platform.platform(), cwd=_cwd, selected_sys_path=_selected_path)

    _file_map = {
        "apps/api/": "/app/",
        "scripts/provider_fingerprint_contract.py": "/opt/scripts/provider_fingerprint_contract.py",
        "scripts/prospective_acquisition/": "/opt/scripts/prospective_acquisition/",
    }
    _current_files = []
    _marker("CURRENT_CHECKOUT_IDENTITY")
    for _rel, _expected_hash in _expected.items():
        _target = None
        if _rel.startswith("apps/api/"):
            _target = "/app/" + _rel
        elif _rel == "scripts/provider_fingerprint_contract.py":
            _target = "/opt/scripts/provider_fingerprint_contract.py"
        elif _rel.startswith("scripts/prospective_acquisition/"):
            _target = "/opt/" + _rel
        else:
            continue
        _path = pathlib.Path(_target).resolve()
        if not _path.is_file():
            _identity["files"].append({"relative_path": _rel, "status": "MISSING"})
            continue
        _actual = hashlib.sha256(_path.read_bytes()).hexdigest()
        _ok = _actual == _expected_hash
        _current_files.append(_ok)
        _identity["files"].append({"relative_path": _rel, "resolved_file": str(_path), "sha256": _actual, "matches_current_checkout": _ok})
    _identity["status"] = "PASS" if _current_files and all(_current_files) and all(f.get("status") != "MISSING" for f in _identity["files"]) else "FAIL"
    if _identity["status"] != "PASS":
        raise _ProbeFailure("CURRENT_CHECKOUT_IDENTITY_FAILED")

    _module_files = {
        "app.api.v1.research_runs":"apps/api/app/api/v1/research_runs.py",
        "app.api.v1.provider_profiles":"apps/api/app/api/v1/provider_profiles.py",
        "app.services.research_runs":"apps/api/app/services/research_runs.py",
        "app.infrastructure.runtime":"apps/api/app/infrastructure/runtime.py",
        "app.infrastructure.db.research_runs":"apps/api/app/infrastructure/db/research_runs.py",
        "app.infrastructure.db.provider_profiles":"apps/api/app/infrastructure/db/provider_profiles.py",
        "app.infrastructure.db.models":"apps/api/app/infrastructure/db/models.py",
        "app.infrastructure.db.postgres":"apps/api/app/infrastructure/db/postgres.py",
        "app.infrastructure.db.research_tools":"apps/api/app/infrastructure/db/research_tools.py",
        "app.services.provider_profiles":"apps/api/app/services/provider_profiles.py",
        "app.security.secrets":"apps/api/app/security/secrets.py",
        "app.core.config":"apps/api/app/core/config.py",
        "app.domain.provider_profiles":"apps/api/app/domain/provider_profiles.py",
        "app.worker.tasks":"apps/api/app/worker/tasks.py",
        "app.worker.dispatcher":"apps/api/app/worker/dispatcher.py",
        "app.worker.celery_app":"apps/api/app/worker/celery_app.py",
        "app.services.research_graph":"apps/api/app/services/research_graph.py",
        "app.services.research_loop":"apps/api/app/services/research_loop.py",
        "app.tools.web_search":"apps/api/app/tools/web_search.py",
        "app.tools.web_reader":"apps/api/app/tools/web_reader.py",
        "app.services.evidence_extractor":"apps/api/app/services/evidence_extractor.py",
        "app.domain.source_context_extraction_v3_3":"apps/api/app/domain/source_context_extraction_v3_3.py",
        "app.services.source_context_extractor_v3_3":"apps/api/app/services/source_context_extractor_v3_3.py",
        "scripts.provider_fingerprint_contract":"scripts/provider_fingerprint_contract.py",
        "scripts.prospective_acquisition.control":"scripts/prospective_acquisition/control.py",
        "scripts.prospective_acquisition.profile":"scripts/prospective_acquisition/profile.py",
        "scripts.prospective_acquisition.schema_validation":"scripts/prospective_acquisition/schema_validation.py",
        "scripts.prospective_acquisition.schema_runtime_probe":"scripts/prospective_acquisition/schema_runtime_probe.py",
    }
    _symbols = {
        "app.infrastructure.db.research_runs":["ResearchRunRepository"],
        "app.infrastructure.db.research_tools":["ResearchToolRepository.record_page"],
        "app.services.research_runs":["ResearchRunService"],
        "app.infrastructure.runtime":["ApplicationRuntime"],
        "app.worker.tasks":["execute_research_run"],
        "app.worker.dispatcher":["dispatch_forever"],
        "app.services.research_graph":["ResearchGraphService"],
        "app.services.research_loop":["ResearchLoopService"],
        "app.services.provider_profiles":["ProviderProfileService"],
        "app.infrastructure.db.provider_profiles":["ProviderProfileRepository"],
        "app.services.evidence_extractor":["EvidenceExtractorService"],
        "app.tools.web_reader":["PublicWebReader"],
        "app.tools.web_search":["SearXNGSearchProvider"],
        "app.domain.source_context_extraction_v3_3":["MeasurementIdentityV33","SOURCE_CONTEXT_EXTRACTION_SCHEMA_V33","SOURCE_CONTEXT_EXTRACTION_DOMAIN_V33","SOURCE_CONTEXT_EXTRACTION_PROMPT_V33"],
        "app.services.source_context_extractor_v3_3":["evaluate_measurement_admissibility_v33"],
        "scripts.prospective_acquisition.control":["ProspectiveAcquisitionHarness"],
        "scripts.prospective_acquisition.profile":["BoundedProspectiveAcquisitionProfile"],
        "scripts.prospective_acquisition.schema_validation":["validator_for_schema","validate_instance"],
        "scripts.prospective_acquisition.schema_runtime_probe":["probe_schema_runtime"],
    }
    _marker("IMPORT_PROBE_STARTED", required_import_count=len(_module_files))
    for _name, _rel in _module_files.items():
        try:
            _mod = importlib.import_module(_name)
            _resolved = pathlib.Path(_mod.__file__).resolve()
            _digest = hashlib.sha256(_resolved.read_bytes()).hexdigest()
            _matched = _digest == _expected.get(_rel)
            _present = all((hasattr(_mod, _sym) if "." not in _sym else hasattr(getattr(_mod, _sym.split(".", 1)[0], None), _sym.split(".", 1)[1])) for _sym in _symbols.get(_name, []))
            _modules.append({"module_name":_name,"status":"PASS" if _matched and _present else "FAIL","resolved_file":str(_resolved),"sha256":_digest,"matches_current_checkout":_matched,"required_symbols_present":_present,"exception_type":None,"safe_error_summary":None})
        except Exception as _exc:
            _modules.append({"module_name":_name,"status":"FAIL","resolved_file":None,"sha256":None,"matches_current_checkout":False,"required_symbols_present":False,"exception_type":type(_exc).__name__,"safe_error_summary":"IMPORT_EXCEPTION:" + type(_exc).__name__})
    _marker("IMPORT_PROBE_COMPLETED", passed=sum(1 for m in _modules if m["status"]=="PASS"), failed=sum(1 for m in _modules if m["status"]!="PASS"))
    if any(m["status"] != "PASS" for m in _modules):
        _stage = "IMPORT_PROBE"
        raise _ProbeFailure("RUNTIME_IMPORT_FAILED")

    _stage = "SCHEMA_RUNTIME_PROBE"
    _marker("SCHEMA_RUNTIME_PROBE_STARTED")
    from scripts.prospective_acquisition.schema_runtime_probe import probe_schema_runtime
    _schema_runtime = probe_schema_runtime(pathlib.Path("/opt/artifacts/v2_1r_d4r_a51/task_manifest_schema.json"))
    _marker("SCHEMA_RUNTIME_PROBE_COMPLETED", status=_schema_runtime["status"])
    if _schema_runtime["status"] != "PASS":
        raise _ProbeFailure("SCHEMA_RUNTIME_DEPENDENCY_OR_VALIDATION_FAILED")

    async def _resolve_provider():
        global _stage, _db, _provider
        _stage = "DB_READ_PROBE"
        _marker("DB_READ_PROBE_STARTED")
        from sqlalchemy import select, text
        from sqlalchemy.engine import make_url
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
        from app.core.config import Settings
        from app.infrastructure.db.models import ProviderProfileRow
        from app.infrastructure.db.provider_profiles import ProviderProfileRepository
        from app.security.secrets import SecretCipher
        from app.services.provider_profiles import ProviderProfileService
        _settings = Settings()
        _db_url = make_url(_settings.database_url)
        _db_safe = {"host":_db_url.host,"port":_db_url.port,"database":_db_url.database}
        try:
            _engine = create_async_engine(_settings.database_url, connect_args={"options":"-c default_transaction_read_only=on"}, pool_pre_ping=True)
        except Exception as _exc:
            _db = {"status":"FAILED","failure_category":"DB_CONNECTION_FAILED","safe_endpoint":_db_safe,"exception_type":type(_exc).__name__}
            raise _ProbeFailure("DB_CONNECTION_FAILED")
        _factory = async_sessionmaker(_engine, expire_on_commit=False)
        try:
            try:
                async with _engine.connect() as _conn:
                    await _conn.execute(text("SELECT 1"))
            except Exception as _exc:
                _db = {"status":"FAILED","failure_category":"DB_CONNECTION_FAILED","safe_endpoint":_db_safe,"exception_type":type(_exc).__name__}
                raise _ProbeFailure("DB_CONNECTION_FAILED")
            try:
                async with _factory() as _session:
                    _readonly = (await _session.execute(text("SHOW transaction_read_only"))).scalar_one()
                    if str(_readonly).lower() not in ("on","true"):
                        _db = {"status":"FAILED","failure_category":"DB_READONLY_CONFIGURATION_FAILED","safe_endpoint":_db_safe}
                        raise _ProbeFailure("DB_READONLY_CONFIGURATION_FAILED")
            except _ProbeFailure:
                raise
            except Exception as _exc:
                _db = {"status":"FAILED","failure_category":"DB_QUERY_FAILED","safe_endpoint":_db_safe,"exception_type":type(_exc).__name__}
                raise _ProbeFailure("DB_QUERY_FAILED")
            _db = {"status":"PASS","read_only_confirmed":True,"safe_endpoint":_db_safe}
            _marker("DB_READ_PROBE_COMPLETED", read_only_confirmed=True)

            _stage = "FROZEN_CONTEXT"
            _marker("FROZEN_CONTEXT_RESOLUTION_STARTED", execution_context_id="8862f286-e707-461e-876f-ec57207802e8")
            import uuid
            _expected_context_id = os.environ.get("OPERATOR_EXECUTION_CONTEXT_ID", "")
            if _expected_context_id != "8862f286-e707-461e-876f-ec57207802e8":
                _provider = {"status":"FAILED","failure_category":"FROZEN_EXECUTION_CONTEXT_INVALID","profiles":[]}
                raise _ProbeFailure("FROZEN_EXECUTION_CONTEXT_INVALID")
            _expected_population_id = "v2_1r_d4r_prospective_population_001"
            _expected_batch_id = "v2_1r_d4r_prospective_population_001_batch_001"
            _expected_owner_source = "AUTHENTICATED_CLIENT_SESSION"
            _expected_policy = "FROZEN_DEFAULT_PROFILE_V1"
            _context_dir = pathlib.Path("/app/artifacts/v2_1r_d4r_a6_closure/execution_contexts")
            _artifact_key = hashlib.sha256(json.dumps([_expected_population_id,_expected_batch_id],ensure_ascii=False,separators=(",",":")).encode("utf-8")).hexdigest()
            _context_path = _context_dir / (_artifact_key + ".json")
            if not _context_path.is_file():
                _provider = {"status":"FAILED","failure_category":"FROZEN_EXECUTION_CONTEXT_ARTIFACT_NOT_FOUND","profiles":[]}
                raise _ProbeFailure("FROZEN_EXECUTION_CONTEXT_ARTIFACT_NOT_FOUND")
            try:
                _context = json.loads(_context_path.read_text(encoding="utf-8"))
                _binding = _validate_frozen_context(_context,_expected_context_id,_expected_population_id,_expected_batch_id)
                _profile_uuid = uuid.UUID(_binding["profile_id"])
                _credential_uuid = uuid.UUID(_binding["credential_version_id"])
            except Exception as _exc:
                _provider = {"status":"FAILED","failure_category":"FROZEN_EXECUTION_CONTEXT_INVALID","profiles":[]}
                raise _ProbeFailure("FROZEN_EXECUTION_CONTEXT_INVALID")

            # The frozen artifact's profile ID is only a lookup key. Recover its
            # internal owner scope, verify the safe fingerprint, and never list
            # or enumerate any other owner scope.
            try:
                async with _factory() as _session:
                    _scope_row = (await _session.execute(
                        select(ProviderProfileRow.owner_hash).where(
                            ProviderProfileRow.id==_profile_uuid,
                            ProviderProfileRow.status=="active",
                            ProviderProfileRow.deleted_at.is_(None),
                        )
                    )).one_or_none()
                if _scope_row is None:
                    raise ValueError("frozen_profile_scope_unavailable")
                _owner_hash = _scope_row.owner_hash
                _owner_scope_fingerprint = hashlib.sha256(_owner_hash.encode("ascii")).hexdigest()
                if _owner_scope_fingerprint != _context["owner_scope_fingerprint"]:
                    raise ValueError("frozen_owner_scope_fingerprint_mismatch")
                _computed_context_hash = _compute_frozen_context_hash(_context,_owner_hash)
                if _computed_context_hash != _context["context_hash"]:
                    raise ValueError("frozen_context_hash_mismatch")
            except Exception as _exc:
                _provider = {"status":"FAILED","failure_category":"FROZEN_EXECUTION_CONTEXT_INVALID","profiles":[],"execution_context":{"status":"FAILED","execution_context_id":_expected_context_id}}
                raise _ProbeFailure("FROZEN_EXECUTION_CONTEXT_INVALID")
            _provider["execution_context"] = _safe_context_projection(_context,"/app/artifacts/v2_1r_d4r_a6_closure/execution_contexts/"+_context_path.name,False)
            _marker("FROZEN_CONTEXT_VALIDATED", execution_context_id=_expected_context_id)

            _stage = "PROVIDER_PROFILE_RESOLUTION"
            _marker("PROVIDER_PROFILE_RESOLUTION_STARTED")
            try:
                _repository = ProviderProfileRepository(_factory)
                _service = ProviderProfileService(_repository,SecretCipher(bytes(32)),allow_insecure_endpoints=_settings.allow_insecure_provider_endpoints)
                _scoped_profiles = await _service.list_profiles(_owner_hash)
                _defaults = [view for view in _scoped_profiles if view.status=="active" and view.is_default]
            except Exception as _exc:
                _provider = {"status":"FAILED","failure_category":"PROFILE_DB_QUERY_FAILED","profiles":[],"exception_type":type(_exc).__name__}
                raise _ProbeFailure("PROVIDER_PROFILE_RESOLUTION_FAILED")
            if len(_defaults) == 0:
                _provider = {"status":"FAILED","failure_category":"NO_DEFAULT_PROFILE_FOR_ACQUISITION_OWNER","profiles":[]}
                raise _ProbeFailure("PROVIDER_PROFILE_RESOLUTION_FAILED")
            if len(_defaults) != 1:
                _provider = {"status":"FAILED","failure_category":"MULTIPLE_DEFAULT_PROFILES_FOR_ACQUISITION_OWNER","profiles":[],"default_profile_count":len(_defaults)}
                raise _ProbeFailure("PROVIDER_PROFILE_RESOLUTION_FAILED")
            _view = _defaults[0]
            if str(_view.profile_id) != _binding["profile_id"]:
                _provider = {"status":"FAILED","failure_category":"EXECUTION_PROVIDER_BINDING_CHANGED","profiles":[]}
                raise _ProbeFailure("EXECUTION_PROVIDER_BINDING_CHANGED")
            _marker("PROVIDER_PROFILE_RESOLUTION_COMPLETED", default_profile_count=1)

            _stage = "PROVIDER_SAFE_FINGERPRINT"
            _marker("SAFE_FINGERPRINT_STARTED")
            try:
                from scripts.provider_fingerprint_contract import safe_provider_fingerprint, FINGERPRINT_CONTRACT_VERSION
                _row, _credential = await _repository.get_active(_owner_hash,_view.profile_id)
                if (
                    _row.owner_hash != _owner_hash
                    or _row.status != "active"
                    or _row.is_default is not True
                    or _row.deleted_at is not None
                    or _credential.revoked_at is not None
                    or _credential.deleted_at is not None
                    or str(_credential.id) != str(_view.credential_version_id)
                    or int(_credential.credential_version) != int(_view.credential_version)
                    or str(_credential.id) != str(_credential_uuid)
                ):
                    raise _ProbeFailure("EXECUTION_PROVIDER_BINDING_CHANGED")
                _safe_input = {"profile_id":str(_view.profile_id),"adapter":_view.adapter_type.value,"model":_view.model,"base_url":_view.base_url,"auth_type":_row.auth_type,"profile_version":_view.version,"credential":{"id":str(_credential.id),"version":_credential.credential_version,"present":True},"non_secret_settings":_row.non_secret_settings}
                _safe_payload,_fingerprint = safe_provider_fingerprint(_safe_input)
                _endpoint = _safe_payload["endpoint"]
                _port = _endpoint["port"]
                _endpoint_family = _endpoint["scheme"]+"://"+_endpoint["host"]+(":"+str(_port) if _port is not None else "")
                _current_binding = {
                    "profile_id":str(_view.profile_id),
                    "adapter":_view.adapter_type.value,
                    "model":_view.model,
                    "endpoint_family":_endpoint_family,
                    "profile_version":_view.version,
                    "credential_version_id":str(_credential.id),
                    "credential_version":_credential.credential_version,
                    "credential_present":True,
                    "fingerprint_contract":FINGERPRINT_CONTRACT_VERSION,
                    "provider_safe_config_fingerprint":_fingerprint,
                }
                if not _provider_binding_matches(_binding,_current_binding):
                    _provider = {"status":"FAILED","failure_category":"EXECUTION_PROVIDER_BINDING_CHANGED","profiles":[],"execution_context":_provider["execution_context"]}
                    raise _ProbeFailure("EXECUTION_PROVIDER_BINDING_CHANGED")
            except Exception as _exc:
                if isinstance(_exc,_ProbeFailure):
                    raise
                _provider = {"status":"FAILED","failure_category":"SAFE_FINGERPRINT_FAILED","profiles":[],"exception_type":type(_exc).__name__}
                raise _ProbeFailure("PROVIDER_SAFE_FINGERPRINT_FAILED")
            _safe_identity = {
                "profile_id":_current_binding["profile_id"],
                "adapter":_current_binding["adapter"],
                "model":_current_binding["model"],
                "endpoint_family":_current_binding["endpoint_family"],
                "profile_version":_current_binding["profile_version"],
                "credential_version_id":_current_binding["credential_version_id"],
                "credential_version":_current_binding["credential_version"],
                "credential_present":_current_binding["credential_present"],
                "fingerprint_contract":_current_binding["fingerprint_contract"],
            }
            _provider["profiles"] = [{"safe_identity":_safe_identity,"provider_safe_config_fingerprint":_fingerprint,"is_default":True,"status":"active","credential_present":True,"profile_updated_at":_view.updated_at.isoformat()}]
            _provider["status"] = "RESOLVED"
            _provider["execution_context"]["provider_binding_match"] = True
            _marker("SAFE_FINGERPRINT_COMPLETED")
            _provider["fingerprint_contract"] = FINGERPRINT_CONTRACT_VERSION
            return _provider
        finally:
            await _engine.dispose()

    _provider = asyncio.run(_resolve_provider())
    _identity["status"] = "PASS"
    _result = _envelope("PASS", python_version=platform.python_version(), executable=sys.executable, platform=platform.platform(), cwd=_cwd, selected_sys_path=_selected_path, modules=_modules, current_checkout_identity=_identity, database=_db, provider=_provider, provider_generation_calls=0, provider_http_count=0, semantic_verifier_calls=0, db_writes=0, holdout_access=0, v33_identity={"schema":"source-context-extraction-schema-v3.3","domain":"source-context-extraction-domain-v3.3","prompt":"source-context-extraction.generic.v3.3","measurement_identity_symbol":"MeasurementIdentityV33","evaluator_symbol":"evaluate_measurement_admissibility_v33","evaluation_executed":False})
    _result["schema_runtime"] = _schema_runtime
    _marker("FINAL_ATTESTATION_READY")
    sys.stdout.write(json.dumps(_result, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()
except BaseException as _exc:
    try:
        _safe_fail(_exc)
    except BaseException:
        pass
    raise SystemExit(21)
