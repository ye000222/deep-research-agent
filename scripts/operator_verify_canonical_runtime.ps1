[CmdletBinding()]
param(
    [switch]$SelfTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$expectedExecutionContextId = '8862f286-e707-461e-876f-ec57207802e8'
$expectedPopulationId = 'v2_1r_d4r_prospective_population_001'
$expectedBatchId = 'v2_1r_d4r_prospective_population_001_batch_001'

function ConvertTo-WindowsArgument([string]$Value) {
    if ($Value.Length -eq 0) { return '""' }
    if ($Value -notmatch '[\s"]') { return $Value }
    $builder = New-Object System.Text.StringBuilder
    [void]$builder.Append([char]34)
    $backslashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq [char]92) { $backslashes++; continue }
        if ($character -eq [char]34) {
            [void]$builder.Append(('\' * (2 * $backslashes + 1)))
            [void]$builder.Append([char]34)
        } else {
            if ($backslashes -gt 0) { [void]$builder.Append(('\' * $backslashes)) }
            [void]$builder.Append($character)
        }
        $backslashes = 0
    }
    if ($backslashes -gt 0) { [void]$builder.Append(('\' * (2 * $backslashes))) }
    [void]$builder.Append([char]34)
    return $builder.ToString()
}

function Invoke-Native([string]$FilePath, [string[]]$Arguments, [int]$TimeoutMilliseconds = 30000) {
    $start = New-Object System.Diagnostics.ProcessStartInfo
    $start.FileName = $FilePath
    $start.Arguments = (($Arguments | ForEach-Object { ConvertTo-WindowsArgument $_ }) -join ' ')
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $start
    $startedAt = [DateTime]::UtcNow
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        if (-not $process.Start()) { throw 'NATIVE_PROCESS_START_FAILED' }
    } catch {
        $_.Exception.Data['NativeInvocationPhase'] = 'PROCESS_START'
        $_.Exception.Data['ArgumentCharacters'] = $start.Arguments.Length
        $process.Dispose()
        throw
    }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $completed = $process.WaitForExit($TimeoutMilliseconds)
    if (-not $completed) {
        $process.Kill()
        $process.WaitForExit()
    }
    $timer.Stop()
    $endedAt = [DateTime]::UtcNow
    return [pscustomobject]@{
        ExitCode = if ($completed) { $process.ExitCode } else { 124 }
        Stdout = $stdoutTask.Result
        Stderr = if ($completed) { $stderrTask.Result } else { 'OPERATOR_VERIFICATION_COMMAND_TIMEOUT' }
        TimedOut = (-not $completed)
        StartedAtUtc = $startedAt.ToString('o')
        EndedAtUtc = $endedAt.ToString('o')
        DurationMs = [long]$timer.ElapsedMilliseconds
    }
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-NativeInvocationFailure([System.Exception]$Exception) {
    $types = @()
    $nativeCode = $null
    $phase = $null
    $argumentCharacters = $null
    $current = $Exception
    while ($null -ne $current -and $types.Count -lt 12) {
        $types += $current.GetType().Name
        if ($current -is [System.ComponentModel.Win32Exception]) { $nativeCode = $current.NativeErrorCode }
        if ($current.Data.Contains('NativeInvocationPhase')) { $phase = $current.Data['NativeInvocationPhase'] }
        if ($current.Data.Contains('ArgumentCharacters')) { $argumentCharacters = $current.Data['ArgumentCharacters'] }
        $current = $current.InnerException
    }
    return [pscustomobject]@{ phase=$phase; exception_types=$types; native_error_code=$nativeCode; argument_characters=$argumentCharacters }
}

function Write-JsonFile([string]$Path, [object]$Value) {
    $json = ConvertTo-Json -InputObject $Value -Depth 40
    [System.IO.File]::WriteAllText($Path, $json + "`n", (New-Object System.Text.UTF8Encoding($false)))
}

function Get-ShortError([string]$Text) {
    if ([string]::IsNullOrWhiteSpace($Text)) { return '' }
    return (Get-SafeDiagnosticText $Text 500) -replace '[\r\n]+', ' '
}

function Get-SafeDiagnosticText([string]$Text, [int]$MaximumLength = 12000) {
    if ([string]::IsNullOrEmpty($Text)) { return '' }
    $safe = $Text -replace '(?i)([a-z][a-z0-9+.-]*://)[^/@\s:]+:[^/@\s]+@', '$1[REDACTED]@'
    $safe = $safe -replace '(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+', '$1[REDACTED]'
    $safe = $safe -replace '(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*', 'Bearer [REDACTED]'
    $safe = $safe -replace '(?i)(password|passwd|pwd|token|secret|api[_-]?key|access[_-]?key)(\s*[=:]\s*)(?:"[^"]*"|''[^'']*''|[^\s,;]+)', '$1$2[REDACTED]'
    $safe = $safe -replace '(?im)^.*(?:DATABASE_URL|MASTER_KEY|CREDENTIAL|API_KEY|AUTHORIZATION|BEARER_TOKEN).*$','[REDACTED_SENSITIVE_DIAGNOSTIC_LINE]'
    if ($safe.Length -gt $MaximumLength) { $safe = $safe.Substring(0, $MaximumLength) + "`n[TRUNCATED]" }
    return $safe
}

function Get-ProbeEnvelope([string]$Stdout) {
    foreach ($line in @($Stdout -split "`r?`n")) {
        $trimmed = $line.Trim()
        if (-not $trimmed.StartsWith('{')) { continue }
        try {
            $item = $trimmed | ConvertFrom-Json -ErrorAction Stop
            if ($item.attestation_schema -in @('d4ra6_runtime_probe.v1','d4ra6_runtime_probe.v2')) { return $item }
        } catch { }
    }
    return $null
}

function Get-SafeProperty([object]$InputObject, [string]$Name) {
    if ($null -eq $InputObject) { return [pscustomobject]@{ state='MISSING'; value=$null } }
    if ($InputObject -is [System.Collections.IDictionary]) {
        if (-not $InputObject.Contains($Name)) { return [pscustomobject]@{ state='MISSING'; value=$null } }
        $value = $InputObject[$Name]
        if ($null -eq $value) { return [pscustomobject]@{ state='NULL'; value=$null } }
        return [pscustomobject]@{ state='PRESENT'; value=$value }
    }
    $property = $InputObject.PSObject.Properties[$Name]
    if (-not $property) { return [pscustomobject]@{ state='MISSING'; value=$null } }
    if ($null -eq $property.Value) { return [pscustomobject]@{ state='NULL'; value=$null } }
    return [pscustomobject]@{ state='PRESENT'; value=$property.Value }
}

function Get-ProbeStageMarkers([string]$Stderr) {
    $markers = @()
    foreach ($line in @($Stderr -split "`r?`n")) {
        if (-not $line.StartsWith('D4RA6_STAGE ')) { continue }
        $json = $line.Substring('D4RA6_STAGE '.Length)
        try { $markers += ,($json | ConvertFrom-Json -ErrorAction Stop) } catch { }
    }
    return $markers
}

function Get-ProbePythonIdentity([object]$Envelope, [object[]]$Markers) {
    $identity = [ordered]@{
        status = 'NOT_ATTESTED'
        python_version = $null
        executable = $null
        platform = $null
        working_directory = $null
        selected_sys_path = @()
    }
    if ($Envelope) {
        foreach ($pair in @(
            @('python_version', 'python_version'),
            @('executable', 'executable'),
            @('platform', 'platform'),
            @('cwd', 'working_directory'),
            @('selected_sys_path', 'selected_sys_path')
        )) {
            $property = $Envelope.PSObject.Properties[$pair[0]]
            if ($property) { $identity[$pair[1]] = $property.Value }
        }
        if ((Get-SafeProperty $Envelope 'status').value -eq 'PASS' -and $identity.python_version -and $identity.executable -and $identity.platform -and $identity.working_directory) {
            $identity.status = 'PASS'
        }
    }
    if (-not $identity.python_version) {
        $pythonMarker = @($Markers | Where-Object { (Get-SafeProperty $_ 'stage').value -eq 'PYTHON_ENV_READY' } | Select-Object -Last 1)
        if ($pythonMarker.Count -gt 0) {
            foreach ($field in @('python_version', 'executable', 'platform', 'cwd', 'selected_sys_path')) {
                $property = $pythonMarker[0].PSObject.Properties[$field]
                if ($property) {
                    $target = if ($field -eq 'cwd') { 'working_directory' } else { $field }
                    $identity[$target] = $property.Value
                }
            }
            if ($identity.status -ne 'PASS' -and $identity.python_version) { $identity.status = 'STAGE_ONLY' }
        }
    }
    return $identity
}

function Get-ProviderFingerprintContractObservation([object]$Envelope) {
    $provider = Get-SafeProperty $Envelope 'provider'
    if ($provider.state -ne 'PRESENT') { return [pscustomobject]@{ state='NOT_REACHED'; value=$null; source_path=$null } }
    $status = Get-SafeProperty $provider.value 'status'
    if ($status.state -ne 'PRESENT' -or $status.value -ne 'RESOLVED') { return [pscustomobject]@{ state='NOT_REACHED'; value=$null; source_path=$null } }
    $schema = Get-SafeProperty $Envelope 'attestation_schema'
    if ($schema.state -eq 'PRESENT' -and $schema.value -eq 'd4ra6_runtime_probe.v2') {
        $contract = Get-SafeProperty $provider.value 'fingerprint_contract'
        $sourcePath = 'provider.fingerprint_contract'
    } elseif ($schema.state -eq 'PRESENT' -and $schema.value -eq 'd4ra6_runtime_probe.v1') {
        $contract = Get-SafeProperty $Envelope 'provider_fingerprint_contract'
        $sourcePath = 'provider_fingerprint_contract'
    } else {
        return [pscustomobject]@{ state='UNSUPPORTED_SCHEMA'; value=$null; source_path=$null }
    }
    return [pscustomobject]@{ state=$contract.state; value=$contract.value; source_path=$sourcePath }
}

function Get-ProviderFingerprintContractFailure([object]$Envelope) {
    $contract = Get-ProviderFingerprintContractObservation $Envelope
    if ($contract.state -eq 'NOT_REACHED') { return $null }
    if ($contract.state -eq 'MISSING') { return 'PROVIDER_FINGERPRINT_CONTRACT_MISSING' }
    if ($contract.state -eq 'NULL') { return 'PROVIDER_FINGERPRINT_CONTRACT_NULL' }
    if ($contract.state -eq 'UNSUPPORTED_SCHEMA') { return 'PROVIDER_FINGERPRINT_CONTRACT_SCHEMA_UNSUPPORTED' }
    if ($contract.value -ne 'provider-safe-config-fingerprint.v1') { return 'PROVIDER_FINGERPRINT_CONTRACT_UNSUPPORTED' }
    return $null
}

function Test-ProbeSuccess([object]$Envelope) {
    $status = Get-SafeProperty $Envelope 'status'
    if ($status.state -ne 'PRESENT' -or $status.value -ne 'PASS') { return $false }
    foreach ($requiredIdentityField in @('python_version', 'executable', 'platform', 'cwd')) {
        $identityProperty = Get-SafeProperty $Envelope $requiredIdentityField
        if ($identityProperty.state -ne 'PRESENT' -or [string]::IsNullOrWhiteSpace([string]$identityProperty.value)) { return $false }
    }
    $checkout = Get-SafeProperty $Envelope 'current_checkout_identity'
    $database = Get-SafeProperty $Envelope 'database'
    $provider = Get-SafeProperty $Envelope 'provider'
    $v33 = Get-SafeProperty $Envelope 'v33_identity'
    $schemaRuntime = Get-SafeProperty $Envelope 'schema_runtime'
    if ($schemaRuntime.state -ne 'PRESENT' -or (Get-SafeProperty $schemaRuntime.value 'status').value -ne 'PASS') { return $false }
    foreach ($schemaGate in @('runtime_dependency_declared','validator_initialized','valid_document_passed','invalid_document_rejected')) {
        if ((Get-SafeProperty $schemaRuntime.value $schemaGate).value -ne $true) { return $false }
    }
    foreach ($schemaCounter in @('provider_generation_calls','provider_http_count','db_writes')) {
        if ((Get-SafeProperty $schemaRuntime.value $schemaCounter).value -ne 0) { return $false }
    }
    if ($checkout.state -ne 'PRESENT' -or (Get-SafeProperty $checkout.value 'status').value -ne 'PASS') { return $false }
    if ($database.state -ne 'PRESENT' -or (Get-SafeProperty $database.value 'status').value -ne 'PASS' -or (Get-SafeProperty $database.value 'read_only_confirmed').value -ne $true) { return $false }
    if ($provider.state -ne 'PRESENT' -or (Get-SafeProperty $provider.value 'status').value -ne 'RESOLVED') { return $false }
    if (Get-ProviderFingerprintContractFailure $Envelope) { return $false }
    foreach ($counterName in @('provider_generation_calls', 'provider_http_count', 'semantic_verifier_calls', 'db_writes', 'holdout_access')) {
        $counter = Get-SafeProperty $Envelope $counterName
        if ($counter.state -ne 'PRESENT' -or $counter.value -ne 0) { return $false }
    }
    if ($v33.state -ne 'PRESENT' -or (Get-SafeProperty $v33.value 'schema').value -ne 'source-context-extraction-schema-v3.3' -or (Get-SafeProperty $v33.value 'domain').value -ne 'source-context-extraction-domain-v3.3' -or (Get-SafeProperty $v33.value 'prompt').value -ne 'source-context-extraction.generic.v3.3' -or (Get-SafeProperty $v33.value 'evaluation_executed').value -ne $false) { return $false }
    $profilesProperty = Get-SafeProperty $provider.value 'profiles'
    if ($profilesProperty.state -ne 'PRESENT' -or @($profilesProperty.value).Count -ne 1) { return $false }
    $profileFingerprint = Get-SafeProperty @($profilesProperty.value)[0] 'provider_safe_config_fingerprint'
    if ($profileFingerprint.state -ne 'PRESENT' -or $profileFingerprint.value -notmatch '^[a-f0-9]{64}$') { return $false }
    $context = Get-SafeProperty $provider.value 'execution_context'
    if ($context.state -ne 'PRESENT' -or (Get-SafeProperty $context.value 'status').value -ne 'PASS' -or (Get-SafeProperty $context.value 'execution_context_id').value -ne $script:expectedExecutionContextId -or (Get-SafeProperty $context.value 'artifact_contract_version').value -ne 'prospective-acquisition-execution-context.v1' -or (Get-SafeProperty $context.value 'owner_scope_source').value -ne 'AUTHENTICATED_CLIENT_SESSION') { return $false }
    if ((Get-SafeProperty $context.value 'context_hash').value -notmatch '^[a-f0-9]{64}$' -or (Get-SafeProperty $context.value 'provider_binding_match').value -ne $true) { return $false }
    $frozenBinding = Get-SafeProperty $context.value 'provider_binding'
    $profile = @($profilesProperty.value)[0]
    if ($frozenBinding.state -ne 'PRESENT' -or (Get-SafeProperty $frozenBinding.value 'provider_safe_config_fingerprint').value -ne $profileFingerprint.value) { return $false }
    foreach ($bindingField in @('profile_id','adapter','model','endpoint_family','profile_version','credential_version_id','credential_version','credential_present','fingerprint_contract')) {
        if ((Get-SafeProperty $profile 'safe_identity').state -ne 'PRESENT') { return $false }
        $safeIdentity = Get-SafeProperty $profile 'safe_identity'
        $safeEndpoint = Get-SafeProperty $safeIdentity.value 'endpoint_family'
        $profileValue = switch ($bindingField) {
            'profile_id' { (Get-SafeProperty $safeIdentity.value 'profile_id').value }
            'adapter' { (Get-SafeProperty $safeIdentity.value 'adapter').value }
            'model' { (Get-SafeProperty $safeIdentity.value 'model').value }
            'endpoint_family' { $safeEndpoint.value }
            'profile_version' { (Get-SafeProperty $safeIdentity.value 'profile_version').value }
            'credential_version_id' { (Get-SafeProperty $safeIdentity.value 'credential_version_id').value }
            'credential_version' { (Get-SafeProperty $safeIdentity.value 'credential_version').value }
            'credential_present' { (Get-SafeProperty $safeIdentity.value 'credential_present').value }
            'fingerprint_contract' { (Get-SafeProperty $safeIdentity.value 'fingerprint_contract').value }
        }
        if ($profileValue -ne (Get-SafeProperty $frozenBinding.value $bindingField).value) { return $false }
    }
    $modulesProperty = Get-SafeProperty $Envelope 'modules'
    if ($modulesProperty.state -ne 'PRESENT' -or @($modulesProperty.value).Count -eq 0) { return $false }
    foreach ($module in @($modulesProperty.value)) {
        if ((Get-SafeProperty $module 'module_name').state -ne 'PRESENT' -or (Get-SafeProperty $module 'resolved_file').state -ne 'PRESENT' -or (Get-SafeProperty $module 'sha256').state -ne 'PRESENT' -or (Get-SafeProperty $module 'status').value -ne 'PASS' -or (Get-SafeProperty $module 'matches_current_checkout').value -ne $true -or (Get-SafeProperty $module 'required_symbols_present').value -ne $true) { return $false }
    }
    return $true
}

function Invoke-OperatorSelfTests {
    $script:expectedExecutionContextId = '8862f286-e707-461e-876f-ec57207802e8'
    $successFixture = '{"attestation_schema":"d4ra6_runtime_probe.v2","status":"PASS","python_version":"3.12.1","executable":"/usr/local/bin/python","platform":"Linux-test","cwd":"/app","selected_sys_path":["/app/apps/api"],"current_checkout_identity":{"status":"PASS"},"database":{"status":"PASS","read_only_confirmed":true},"provider":{"status":"RESOLVED","fingerprint_contract":"provider-safe-config-fingerprint.v1","profiles":[{"safe_identity":{"profile_id":"p1","adapter":"openai_compatible_chat","model":"m1","endpoint_family":"https://example.test","profile_version":2,"credential_version_id":"c1","credential_version":3,"credential_present":true,"fingerprint_contract":"provider-safe-config-fingerprint.v1"},"provider_safe_config_fingerprint":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}],"execution_context":{"status":"PASS","execution_context_id":"8862f286-e707-461e-876f-ec57207802e8","artifact_contract_version":"prospective-acquisition-execution-context.v1","owner_scope_source":"AUTHENTICATED_CLIENT_SESSION","context_hash":"abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789","provider_binding_match":true,"provider_binding":{"profile_id":"p1","adapter":"openai_compatible_chat","model":"m1","endpoint_family":"https://example.test","profile_version":2,"credential_version_id":"c1","credential_version":3,"credential_present":true,"fingerprint_contract":"provider-safe-config-fingerprint.v1","provider_safe_config_fingerprint":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}}},"v33_identity":{"schema":"source-context-extraction-schema-v3.3","domain":"source-context-extraction-domain-v3.3","prompt":"source-context-extraction.generic.v3.3","evaluation_executed":false},"modules":[{"module_name":"app.example","status":"PASS","resolved_file":"/app/app/example.py","sha256":"abcdef","matches_current_checkout":true,"required_symbols_present":true}],"provider_generation_calls":0,"provider_http_count":0,"semantic_verifier_calls":0,"db_writes":0,"holdout_access":0}'
    $success = Get-ProbeEnvelope $successFixture
    $success | Add-Member -NotePropertyName schema_runtime -NotePropertyValue ([pscustomobject]@{status='PASS';runtime_dependency_declared=$true;validator_initialized=$true;valid_document_passed=$true;invalid_document_rejected=$true;provider_generation_calls=0;provider_http_count=0;db_writes=0})
    if (-not (Test-ProbeSuccess $success)) { throw 'SELF_TEST_SUCCESS_GATE_FAILED' }
    $missingSchemaRuntime = Get-ProbeEnvelope $successFixture
    if (Test-ProbeSuccess $missingSchemaRuntime) { throw 'SELF_TEST_MISSING_SCHEMA_RUNTIME_FAIL_CLOSED' }
    $successFixture = $success | ConvertTo-Json -Depth 40 -Compress
    $failedSchemaRuntime = $success | ConvertTo-Json -Depth 40 | ConvertFrom-Json
    $failedSchemaRuntime.schema_runtime.status = 'FAIL'
    if (Test-ProbeSuccess $failedSchemaRuntime) { throw 'SELF_TEST_FAILED_SCHEMA_RUNTIME_FAIL_CLOSED' }
    $missingFrozenContext = Get-ProbeEnvelope $successFixture
    $missingFrozenContext.provider.PSObject.Properties.Remove('execution_context')
    if (Test-ProbeSuccess $missingFrozenContext) { throw 'SELF_TEST_MISSING_FROZEN_CONTEXT_FAIL_CLOSED' }
    $wrongFrozenContext = Get-ProbeEnvelope $successFixture
    $wrongFrozenContext.provider.execution_context.execution_context_id = '00000000-0000-0000-0000-000000000000'
    if (Test-ProbeSuccess $wrongFrozenContext) { throw 'SELF_TEST_WRONG_FROZEN_CONTEXT_FAIL_CLOSED' }
    $providerDrift = Get-ProbeEnvelope $successFixture
    $providerDrift.provider.profiles[0].safe_identity.credential_version = 5
    if (Test-ProbeSuccess $providerDrift) { throw 'SELF_TEST_PROVIDER_BINDING_DRIFT_FAIL_CLOSED' }
    $failureFixture = '{"attestation_schema":"d4ra6_runtime_probe.v2","status":"FAILED","failed_stage":"IMPORT_PROBE","exception_type":"ModuleNotFoundError","safe_message":"RUNTIME_IMPORT_FAILED"}'
    $failure = Get-ProbeEnvelope $failureFixture
    if (-not $failure -or $failure.failed_stage -ne 'IMPORT_PROBE') { throw 'SELF_TEST_FAILURE_ENVELOPE_PARSE_FAILED' }
    $stderrFixture = "D4RA6_STAGE {`"stage`":`"IMPORT_PROBE_STARTED`"}`nModuleNotFoundError: safe module absent"
    $markers = @(Get-ProbeStageMarkers $stderrFixture)
    if ($markers.Count -ne 1 -or $markers[0].stage -ne 'IMPORT_PROBE_STARTED') { throw 'SELF_TEST_STAGE_MARKER_PARSE_FAILED' }
    $emptyMarkers = @(Get-ProbeStageMarkers 'ordinary stderr without stage marker')
    if ($emptyMarkers.Count -ne 0) { throw 'SELF_TEST_EMPTY_STAGE_MARKERS_FAILED' }
    if (Get-ProbeEnvelope $stderrFixture) { throw 'SELF_TEST_STDOUT_STDERR_PROVENANCE_FAILED' }
    $pythonMarkerFixture = '{"stage":"PYTHON_ENV_READY","python_version":"3.12.1","executable":"/usr/local/bin/python","platform":"Linux-test","cwd":"/app","selected_sys_path":["/app/apps/api"]}' | ConvertFrom-Json
    $failureIdentityFixture = Get-ProbePythonIdentity $failure @($pythonMarkerFixture)
    if ($failureIdentityFixture.status -ne 'STAGE_ONLY' -or $failureIdentityFixture.python_version -ne '3.12.1') { throw 'SELF_TEST_FAILURE_PYTHON_IDENTITY_DIAGNOSTIC_FAILED' }
    $missingIdentityFixture = Get-ProbePythonIdentity $failure @()
    if ($missingIdentityFixture.status -ne 'NOT_ATTESTED' -or $null -ne $missingIdentityFixture.python_version) { throw 'SELF_TEST_MISSING_PYTHON_IDENTITY_FAILED' }
    $presentFixture = Get-ProviderFingerprintContractObservation (Get-ProbeEnvelope $successFixture)
    if ($presentFixture.state -ne 'PRESENT' -or $presentFixture.value -ne 'provider-safe-config-fingerprint.v1' -or $presentFixture.source_path -ne 'provider.fingerprint_contract') { throw 'SELF_TEST_PRESENT_PROPERTY_FAILED' }
    $failureProviderFixture = [pscustomobject]@{ status='FAILED'; profiles=@() }
    $missingFingerprintContract = Get-SafeProperty $failureProviderFixture 'fingerprint_contract'
    if ($missingFingerprintContract.state -ne 'MISSING' -or $null -ne $missingFingerprintContract.value) { throw 'SELF_TEST_OPTIONAL_FINGERPRINT_CONTRACT_FAILED' }
    $nullFingerprintContract = Get-SafeProperty ([pscustomobject]@{fingerprint_contract=$null}) 'fingerprint_contract'
    if ($nullFingerprintContract.state -ne 'NULL' -or $null -ne $nullFingerprintContract.value) { throw 'SELF_TEST_NULL_PROPERTY_FAILED' }
    if ((Get-SafeProperty $failureProviderFixture 'status').state -ne 'PRESENT') { throw 'SELF_TEST_REQUIRED_PROPERTY_PRESENT_FAILED' }
    if ((Get-SafeProperty $failureProviderFixture 'not_a_real_field').state -ne 'MISSING') { throw 'SELF_TEST_REQUIRED_PROPERTY_MISSING_FAILED' }
    if (Test-ProbeSuccess $failureProviderFixture) { throw 'SELF_TEST_REQUIRED_PROBE_FIELD_FAIL_CLOSED' }
    $olderShape = '{"attestation_schema":"d4ra6_runtime_probe.v1","status":"FAILED","provider":{"status":"FAILED","profiles":[]}}' | ConvertFrom-Json
    if ((Get-SafeProperty $olderShape.provider 'fingerprint_contract').state -ne 'MISSING' -or (Test-ProbeSuccess $olderShape)) { throw 'SELF_TEST_OLDER_SHAPE_FAIL_CLOSED' }
    $incompleteSuccessFixture = Get-ProbeEnvelope $successFixture
    $incompleteSuccessFixture.PSObject.Properties.Remove('python_version')
    if (Test-ProbeSuccess $incompleteSuccessFixture) { throw 'SELF_TEST_REQUIRED_PYTHON_IDENTITY_GATE_FAILED' }
    $missingRequiredContract = Get-ProbeEnvelope $successFixture
    $missingRequiredContract.provider.PSObject.Properties.Remove('fingerprint_contract')
    if ((Get-ProviderFingerprintContractFailure $missingRequiredContract) -ne 'PROVIDER_FINGERPRINT_CONTRACT_MISSING' -or (Test-ProbeSuccess $missingRequiredContract)) { throw 'SELF_TEST_REQUIRED_FINGERPRINT_CONTRACT_FAIL_CLOSED' }
    $notReachedContract = Get-ProbeEnvelope $failureFixture
    if ((Get-ProviderFingerprintContractObservation $notReachedContract).state -ne 'NOT_REACHED' -or (Get-ProviderFingerprintContractFailure $notReachedContract)) { throw 'SELF_TEST_FINGERPRINT_CONTRACT_NOT_REACHED_STATE_FAILED' }
    $legacyShape = '{"attestation_schema":"d4ra6_runtime_probe.v1","status":"FAILED","provider_fingerprint_contract":"provider-safe-config-fingerprint.v1","provider":{"status":"RESOLVED","profiles":[]}}' | ConvertFrom-Json
    $legacyContract = Get-ProviderFingerprintContractObservation $legacyShape
    if ($legacyContract.state -ne 'PRESENT' -or $legacyContract.source_path -ne 'provider_fingerprint_contract') { throw 'SELF_TEST_LEGACY_SCHEMA_CONTRACT_PATH_FAILED' }
    if (Test-ProbeSuccess (Get-ProbeEnvelope ($successFixture -replace '"provider_http_count":0','"provider_http_count":1'))) { throw 'SELF_TEST_FAIL_CLOSED_GATE_FAILED' }
    $redacted = Get-SafeDiagnosticText 'Authorization: Bearer secret123 password=hunter2 postgresql://user:pw@db/name'
    if ($redacted -match 'secret123|hunter2|user:pw') { throw 'SELF_TEST_DIAGNOSTIC_REDACTION_FAILED' }
    return 'OPERATOR_PROBE_FIXTURE_TESTS_PASS'
}

if ($SelfTest) {
    try { Write-Output (Invoke-OperatorSelfTests); exit 0 }
    catch { Write-Error (($_.Exception.Message) + ' | ' + $_.ScriptStackTrace); exit 1 }
}

function Get-GitSnapshot([string]$GitExe, [string]$Root) {
    $head = Invoke-Native $GitExe @('-C', $Root, 'rev-parse', 'HEAD')
    if ($head.ExitCode -ne 0) { throw 'GIT_HEAD_UNAVAILABLE' }
    $status = Invoke-Native $GitExe @('-C', $Root, 'status', '--porcelain=v1')
    if ($status.ExitCode -ne 0) { throw 'GIT_STATUS_UNAVAILABLE' }
    $lines = @($status.Stdout -split "`r?`n" | Where-Object { $_.Length -gt 0 })
    $staged = 0; $unstaged = 0; $untracked = 0
    foreach ($line in $lines) {
        if ($line.Length -lt 2) { continue }
        if ($line.Substring(0, 2) -eq '??') { $untracked++; continue }
        if ($line[0] -ne ' ') { $staged++ }
        if ($line[1] -ne ' ') { $unstaged++ }
    }
    return [pscustomobject]@{
        Head = $head.Stdout.Trim()
        StatusSummary = [pscustomobject]@{ entries = $lines.Count; staged = $staged; unstaged = $unstaged; untracked = $untracked }
    }
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$invocationDirectory = (Get-Location).Path.TrimEnd('\')
$outputRoot = Join-Path $repoRoot 'artifacts\v2_1r_d4r_a6_operator'
$outputDirectory = Join-Path $outputRoot ('run_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '_' + [Guid]::NewGuid().ToString('N').Substring(0, 8))
$dockerHostAttestation = $null
$runtimeAttestation = $null
$providerAttestation = $null
$runtimeStatus = 'NOT_STARTED'
$providerStatus = 'NOT_RESOLVED'
$failureCategory = $null
$ephemeralName = $null
$cleanupResult = 'NOT_REQUIRED'
$dockerExe = $null
$gitExe = $null
$dockerPrechecks = $null
$hostIdentity = $null
$probeRun = $null
$probeResult = $null
$probeMarkers = @()
$probeImports = @()
$probeStart = $null
$probeEnd = $null
$probeDurationMs = $null
$probeStdoutSafe = ''
$probeStderrSafe = ''
$probeFailureStage = $null
$currentCheckoutStatus = 'NOT_EVALUATED'
$probeInvocationExceptionType = $null
$probeInvocationFailure = $null
$hostProbeValidationFailure = $null

$runtimeFiles = @(
    'docker-compose.yml',
    'infra/docker/api.Dockerfile',
    'apps/api/app/api/v1/research_runs.py',
    'apps/api/app/api/v1/provider_profiles.py',
    'apps/api/app/services/research_runs.py',
    'apps/api/app/infrastructure/runtime.py',
    'apps/api/app/infrastructure/db/research_runs.py',
    'apps/api/app/infrastructure/db/provider_profiles.py',
    'apps/api/app/infrastructure/db/models.py',
    'apps/api/app/infrastructure/db/postgres.py',
    'apps/api/app/infrastructure/db/research_tools.py',
    'apps/api/app/services/provider_profiles.py',
    'apps/api/app/security/secrets.py',
    'apps/api/app/core/config.py',
    'apps/api/app/domain/provider_profiles.py',
    'apps/api/app/worker/tasks.py',
    'apps/api/app/worker/dispatcher.py',
    'apps/api/app/worker/celery_app.py',
    'apps/api/app/services/research_graph.py',
    'apps/api/app/services/research_loop.py',
    'apps/api/app/tools/web_search.py',
    'apps/api/app/tools/web_reader.py',
    'apps/api/app/services/evidence_extractor.py',
    'apps/api/app/domain/source_context_extraction_v3_3.py',
    'apps/api/app/services/source_context_extractor_v3_3.py',
    'scripts/prospective_acquisition/control.py',
    'scripts/prospective_acquisition/profile.py',
    'scripts/prospective_acquisition/schema_validation.py',
    'scripts/prospective_acquisition/schema_runtime_probe.py',
    'scripts/prospective_acquisition/canonical_runtime_probe.py',
    'scripts/provider_fingerprint_contract.py',
    'scripts/operator_verify_canonical_runtime.ps1'
)

if ($invocationDirectory -ine $repoRoot) {
    Write-Error 'Run this verifier from D:\agentProject; no files or Docker resources were changed.'
    exit 2
}
if (-not (Test-Path -LiteralPath (Join-Path $repoRoot '.git')) -or -not (Test-Path -LiteralPath (Join-Path $repoRoot 'docker-compose.yml'))) {
    Write-Error 'Expected repository metadata or docker-compose.yml is missing; no Docker command was run.'
    exit 2
}
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null

try {
    $dockerCommand = Get-Command docker -ErrorAction Stop | Select-Object -First 1
    $dockerExe = $dockerCommand.Source
    $gitCommand = Get-Command git -ErrorAction Stop | Select-Object -First 1
    $gitExe = $gitCommand.Source
    $before = Get-GitSnapshot $gitExe $repoRoot
    $sourceHashes = [ordered]@{}
    foreach ($relative in $runtimeFiles) {
        $absolute = Join-Path $repoRoot $relative
        if (-not (Test-Path -LiteralPath $absolute -PathType Leaf)) { throw "VERIFICATION_FILE_MISSING:$relative" }
        $sourceHashes[$relative.Replace('\', '/')] = Get-Sha256 $absolute
    }

    $whoami = Invoke-Native (Join-Path $env:SystemRoot 'System32\whoami.exe') @()
    if ($whoami.ExitCode -ne 0) { throw 'HOST_IDENTITY_UNAVAILABLE' }
    $hostIdentity = $whoami.Stdout.Trim()

    $dockerVersion = Invoke-Native $dockerExe @('version')
    $context = Invoke-Native $dockerExe @('context', 'show')
    $dockerPs = Invoke-Native $dockerExe @('ps')
    $composeConfig = Invoke-Native $dockerExe @('compose', 'config', '--format', 'json')

    $versionClient = [regex]::Match($dockerVersion.Stdout, '(?m)^ Version:\s*(.+)$').Groups[1].Value.Trim()
    $serverStart = $dockerVersion.Stdout.IndexOf("Server:", [StringComparison]::Ordinal)
    $versionServer = ''
    if ($serverStart -ge 0) {
        $serverSection = $dockerVersion.Stdout.Substring($serverStart)
        $versionServer = [regex]::Match($serverSection, '(?m)^\s+Version:\s*(.+)$').Groups[1].Value.Trim()
    }
    $dockerPrechecks = [ordered]@{
        docker_version = @{ exit_code=$dockerVersion.ExitCode; client_version=$versionClient; server_version=$versionServer; stderr_summary=(Get-ShortError $dockerVersion.Stderr) }
        docker_context_show = @{ exit_code=$context.ExitCode; active_context=$context.Stdout.Trim(); stderr_summary=(Get-ShortError $context.Stderr) }
        docker_ps = @{ exit_code=$dockerPs.ExitCode; running_container_count=[Math]::Max(0,(@($dockerPs.Stdout -split "`r?`n" | Where-Object { $_.Trim() }).Count - 1)); stderr_summary=(Get-ShortError $dockerPs.Stderr) }
        docker_compose_config = @{ exit_code=$composeConfig.ExitCode; valid_json=($composeConfig.ExitCode -eq 0); stderr_present=(-not [string]::IsNullOrWhiteSpace($composeConfig.Stderr)) }
    }

    $composeServices = @()
    $apiImage = $null
    $sourceMountVerified = $false
    $artifactVolumeVerified = $false
    if ($composeConfig.ExitCode -eq 0) {
        try {
            $composeObject = $composeConfig.Stdout | ConvertFrom-Json -ErrorAction Stop
            $composeServices = @($composeObject.services.PSObject.Properties.Name)
            $apiService = $composeObject.services.api
            $apiImage = [string]$apiService.image
            $expectedMount = [System.IO.Path]::GetFullPath((Join-Path $repoRoot 'apps\api')).TrimEnd('\')
            foreach ($mount in @($apiService.volumes)) {
                if ($mount.type -eq 'bind' -and $mount.target -eq '/app/apps/api') {
                    $actualMount = [System.IO.Path]::GetFullPath([string]$mount.source).TrimEnd('\')
                    if ($actualMount -ieq $expectedMount -and $mount.read_only -eq $true) { $sourceMountVerified = $true }
                }
                if ($mount.type -eq 'volume' -and $mount.target -eq '/app/artifacts') { $artifactVolumeVerified = $true }
            }
        } catch { throw 'COMPOSE_CONFIG_NOT_PARSEABLE_OR_API_MOUNT_UNVERIFIED' }
    }
    if ($dockerVersion.ExitCode -ne 0 -or $context.ExitCode -ne 0 -or $dockerPs.ExitCode -ne 0 -or $composeConfig.ExitCode -ne 0 -or -not $versionClient -or -not $versionServer) {
        $failureCategory = 'DOCKER_HOST_PRECHECK_FAILED'
        throw $failureCategory
    }
    if (-not $apiImage -or -not $sourceMountVerified -or -not $artifactVolumeVerified) {
        $failureCategory = if (-not $sourceMountVerified) { 'CURRENT_CHECKOUT_READONLY_MOUNT_NOT_PROVEN' } else { 'FROZEN_CONTEXT_ARTIFACT_VOLUME_NOT_PROVEN' }
        throw $failureCategory
    }

    # Inspect only an already-local image: never pull, build, or replace images.
    $imageInspect = Invoke-Native $dockerExe @('image', 'inspect', $apiImage)
    if ($imageInspect.ExitCode -ne 0) {
        $failureCategory = 'LOCAL_API_IMAGE_UNAVAILABLE_NO_PULL_OR_BUILD_ATTEMPTED'
        throw $failureCategory
    }
    $imageObject = @($imageInspect.Stdout | ConvertFrom-Json -ErrorAction Stop)[0]
    $imageId = [string]$imageObject.Id

    $expectedHashJson = ConvertTo-Json -InputObject $sourceHashes -Compress -Depth 10
    $expectedHashB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($expectedHashJson))

    $probeArgument = '/opt/scripts/prospective_acquisition/canonical_runtime_probe.py'
    # Execute the unchanged probe from the existing read-only script mount.
    # Large Python source must not enter Windows CreateProcess command arguments.
    $ephemeralName = 'd4ra6-operator-' + [Guid]::NewGuid().ToString('N').Substring(0, 12)
    $runtimeStatus = 'RUNNING_EPHEMERAL_CURRENT_SOURCE_PROBE'
    $runArgs = @(
        'compose','run','--no-deps','--name',$ephemeralName,
        '-e',"OPERATOR_EXPECTED_HASHES_B64=$expectedHashB64",
        '-e',"OPERATOR_EXECUTION_CONTEXT_ID=$expectedExecutionContextId",
        '-v',((Join-Path $repoRoot 'scripts\prospective_acquisition') + ':/opt/scripts/prospective_acquisition:ro'),
        '-v',((Join-Path $repoRoot 'scripts\provider_fingerprint_contract.py') + ':/opt/scripts/provider_fingerprint_contract.py:ro'),
        '-v',((Join-Path $repoRoot 'artifacts\v2_1r_d4r_a51') + ':/opt/artifacts/v2_1r_d4r_a51:ro'),
        '--entrypoint','python','api',$probeArgument
    )
    $probeStart = [DateTime]::UtcNow
    try { $probeRun = Invoke-Native $dockerExe $runArgs 240000 }
    catch {
        $probeEnd = [DateTime]::UtcNow
        $probeDurationMs = [long]($probeEnd - $probeStart).TotalMilliseconds
        $failureCategory = 'EPHEMERAL_CONTAINER_COMMAND_FAILED'
        $runtimeStatus = 'FAIL'
        $probeFailureStage = 'COMPOSE_RUN_INVOCATION'
        $probeInvocationExceptionType = $_.Exception.GetType().Name
        $probeInvocationFailure = Get-NativeInvocationFailure $_.Exception
        throw $failureCategory
    }
    $probeStart = [DateTime]::Parse($probeRun.StartedAtUtc).ToUniversalTime()
    $probeEnd = [DateTime]::Parse($probeRun.EndedAtUtc).ToUniversalTime()
    $probeDurationMs = $probeRun.DurationMs
    $probeStdoutSafe = Get-SafeDiagnosticText $probeRun.Stdout
    $probeStderrSafe = Get-SafeDiagnosticText $probeRun.Stderr
    $probeMarkers = @(Get-ProbeStageMarkers $probeRun.Stderr)
    $probeResult = Get-ProbeEnvelope $probeRun.Stdout
    if ($probeResult) {
        $importsProperty = Get-SafeProperty $probeResult 'imports'
        $modulesProperty = Get-SafeProperty $probeResult 'modules'
        if ($importsProperty.state -eq 'PRESENT') { $probeImports = @($importsProperty.value) }
        elseif ($modulesProperty.state -eq 'PRESENT') { $probeImports = @($modulesProperty.value) }
        $checkoutProperty = Get-SafeProperty $probeResult 'current_checkout_identity'
        if ($checkoutProperty.state -eq 'PRESENT') {
            $checkoutStatusProperty = Get-SafeProperty $checkoutProperty.value 'status'
            if ($checkoutStatusProperty.state -eq 'PRESENT') { $currentCheckoutStatus = [string]$checkoutStatusProperty.value }
        }
        $failedStageProperty = Get-SafeProperty $probeResult 'failed_stage'
        if ($failedStageProperty.state -eq 'PRESENT') { $probeFailureStage = [string]$failedStageProperty.value }
    }
    if (-not $probeResult) {
        $lastStage = if ($probeMarkers.Count -gt 0) { [string]$probeMarkers[-1].stage } else { $null }
        if ($probeRun.TimedOut) { $failureCategory = 'EPHEMERAL_CONTAINER_COMMAND_FAILED' }
        elseif ($probeRun.ExitCode -ne 0 -and $lastStage -eq 'IMPORT_PROBE_STARTED') { $failureCategory = 'RUNTIME_IMPORT_FAILED' }
        elseif ($probeRun.ExitCode -ne 0 -and $lastStage -like 'DB_*') { $failureCategory = 'DATABASE_READ_PROBE_FAILED' }
        elseif ($probeRun.ExitCode -ne 0 -and $lastStage -like 'PROVIDER_*') { $failureCategory = 'PROVIDER_PROFILE_RESOLUTION_FAILED' }
        elseif ($probeRun.ExitCode -ne 0) { $failureCategory = 'PYTHON_PROBE_STARTUP_FAILED' }
        else { $failureCategory = 'PROBE_OUTPUT_PARSE_FAILED' }
        $probeFailureStage = if ($lastStage) { $lastStage } else { 'NO_STAGE_MARKER' }
        $runtimeStatus = 'BLOCKED_' + $failureCategory
        throw $failureCategory
    }
    $providerProperty = Get-SafeProperty $probeResult 'provider'
    $fingerprintContractProperty = Get-ProviderFingerprintContractObservation $probeResult
    $fingerprintContractFailure = Get-ProviderFingerprintContractFailure $probeResult
    if ($fingerprintContractFailure) {
        $failureCategory = $fingerprintContractFailure
        $probeFailureStage = 'PROVIDER_SAFE_FINGERPRINT'
        $runtimeStatus = 'BLOCKED_' + $failureCategory
        $providerStatus = 'FAILED'
        $hostProbeValidationFailure = [ordered]@{status='FAILED';failed_stage=$probeFailureStage;failure_category=$failureCategory;provider_fingerprint_contract_state=$fingerprintContractProperty.state;provider_fingerprint_contract_source=$fingerprintContractProperty.source_path}
        throw $failureCategory
    }
    $envelopeStatus = Get-SafeProperty $probeResult 'status'
    if ($envelopeStatus.state -ne 'PRESENT' -or $envelopeStatus.value -ne 'PASS' -or $probeRun.ExitCode -ne 0 -or -not (Test-ProbeSuccess $probeResult)) {
        $failedStage = Get-SafeProperty $probeResult 'failed_stage'
        if ($failedStage.state -eq 'PRESENT') { $probeFailureStage = [string]$failedStage.value } else { $probeFailureStage = 'SUCCESS_REQUIREMENT_VALIDATION' }
        $checkoutProperty = Get-SafeProperty $probeResult 'current_checkout_identity'
        $checkoutStatusProperty = if ($checkoutProperty.state -eq 'PRESENT') { Get-SafeProperty $checkoutProperty.value 'status' } else { $null }
        $currentCheckoutStatus = if ($checkoutStatusProperty -and $checkoutStatusProperty.state -eq 'PRESENT') { [string]$checkoutStatusProperty.value } else { 'NOT_PROVEN' }
        if ($providerProperty.state -eq 'PRESENT') {
            $profileResolutionStatus = Get-SafeProperty $providerProperty.value 'status'
            if ($profileResolutionStatus.state -eq 'PRESENT') { $providerStatus = [string]$profileResolutionStatus.value }
        }
        $databaseProperty = Get-SafeProperty $probeResult 'database'
        $databaseFailure = if ($databaseProperty.state -eq 'PRESENT') { Get-SafeProperty $databaseProperty.value 'failure_category' } else { $null }
        $providerFailure = if ($providerProperty.state -eq 'PRESENT') { Get-SafeProperty $providerProperty.value 'failure_category' } else { $null }
        $safeMessage = Get-SafeProperty $probeResult 'safe_message'
        if ($databaseFailure -and $databaseFailure.state -eq 'PRESENT') { $failureCategory = [string]$databaseFailure.value }
        elseif ($providerFailure -and $providerFailure.state -eq 'PRESENT') { $failureCategory = [string]$providerFailure.value }
        elseif ($safeMessage.state -eq 'PRESENT') { $failureCategory = [string]$safeMessage.value }
        else { $failureCategory = 'CURRENT_CHECKOUT_IDENTITY_FAILED' }
        $runtimeStatus = 'BLOCKED_' + $failureCategory
        throw $failureCategory
    }
    $runtimeStatus = 'PASS'
    $providerStatus = 'RESOLVED'
    $after = Get-GitSnapshot $gitExe $repoRoot
    if ($after.Head -ne $before.Head) { throw 'GIT_HEAD_CHANGED_DURING_ATTESTATION' }
    foreach ($relative in $runtimeFiles) {
        $currentHash = Get-Sha256 (Join-Path $repoRoot $relative)
        if ($currentHash -ne $sourceHashes[$relative.Replace('\', '/')]) { throw "VERIFICATION_SOURCE_CHANGED:$relative" }
    }

        $runtimeFingerprintPayload = [ordered]@{
        contract_version = 'canonical-runtime-fingerprint.v1'
        git_head = $before.Head
        source_file_sha256 = $sourceHashes
        compose_image_id = $imageId
        schema_runtime = (Get-SafeProperty $probeResult 'schema_runtime').value
        python_version = $probeResult.python_version
        imported_component_identities = @($probeResult.modules | ForEach-Object { [ordered]@{ module=$_.module_name; file=$_.resolved_file; sha256=$_.sha256 } })
        execution_context_id = $probeResult.provider.execution_context.execution_context_id
        execution_context_hash = $probeResult.provider.execution_context.context_hash
        provider_safe_config_fingerprint = $probeResult.provider.profiles[0].provider_safe_config_fingerprint
    }
    $runtimeFingerprintCanonical = ConvertTo-Json -InputObject $runtimeFingerprintPayload -Depth 40 -Compress
    $runtimeHasher = [System.Security.Cryptography.SHA256]::Create()
    $runtimeFingerprintBytes = $runtimeHasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($runtimeFingerprintCanonical))
    $runtimeHasher.Dispose()
    $runtimeFingerprint = ([BitConverter]::ToString($runtimeFingerprintBytes)).Replace('-','').ToLowerInvariant()

    $dockerHostAttestation = [ordered]@{
        artifact_version = 'v2_1r_d4r_a6_operator.docker_host.v1'
        produced_by = 'operator_host_powershell'
        host_identity = $hostIdentity
        repository_path = $repoRoot
        timestamp_utc = [DateTime]::UtcNow.ToString('o')
        git_head = $before.Head
        git_status_summary = $before.StatusSummary
        commands = [ordered]@{
            docker_version = $dockerPrechecks.docker_version
            docker_context_show = $dockerPrechecks.docker_context_show
            docker_ps = $dockerPrechecks.docker_ps
            docker_compose_config = @{ exit_code=$composeConfig.ExitCode; valid_json=($composeConfig.ExitCode -eq 0); service_names=$composeServices; api_source_mount_current_checkout_read_only=$sourceMountVerified; stderr_present=(-not [string]::IsNullOrWhiteSpace($composeConfig.Stderr)) }
        }
        ephemeral_container = @{ name=$ephemeralName; compose_service='api'; compose_run_exit_code=$probeRun.ExitCode; removed_by_operator_script=$true; cleanup='--rm not used; exact uniquely named container removed in finally' }
        docker_calls_by_codex = 0
        provider_generation_calls = 0
        provider_http_count = 0
        semantic_verifier_calls = 0
        db_writes = 0
        real_acquisition = 0
        holdout_access = 0
    }
    $runtimeAttestation = [ordered]@{
        artifact_version = 'v2_1r_d4r_a6_operator.canonical_runtime.v1'
        status = $runtimeStatus
        repository_path = $repoRoot
        git_head = $before.Head
        git_status_summary = $before.StatusSummary
        source_file_sha256 = $sourceHashes
        image = @{ reference=$apiImage; image_id=$imageId; freshly_built=$false; current_checkout_source_read_only_mount_verified=$sourceMountVerified }
        schema_runtime = (Get-SafeProperty $probeResult 'schema_runtime').value
        runtime = @{ python_version=$probeResult.python_version; executable=$probeResult.executable; platform=$probeResult.platform; working_directory=$probeResult.cwd; selected_sys_path=$probeResult.selected_sys_path; ephemeral_container_name=$ephemeralName }
        canonical_runtime_fingerprint_contract = 'canonical-runtime-fingerprint.v1'
        canonical_runtime_fingerprint = $runtimeFingerprint
        execution_context_id = $probeResult.provider.execution_context.execution_context_id
        execution_context_hash = $probeResult.provider.execution_context.context_hash
        imported_components = $probeResult.modules
        v3_3_identity = $probeResult.v33_identity
        provider_profile_resolution_status = $providerStatus
        provider_generation_calls = 0
        provider_http_count = 0
        db_writes = 0
        real_claim_creation = 0
        holdout_access = 0
    }
    $providerAttestation = [ordered]@{
        artifact_version = 'v2_1r_d4r_a6_operator.provider_readiness.v1'
        status = $providerStatus
        execution_context_id = $probeResult.provider.execution_context.execution_context_id
        execution_context_hash = $probeResult.provider.execution_context.context_hash
        execution_context_artifact = $probeResult.provider.execution_context.artifact_path
        owner_scope_source = $probeResult.provider.execution_context.owner_scope_source
        owner_scope_fingerprint = $probeResult.provider.execution_context.owner_scope_fingerprint
        frozen_provider_binding = $probeResult.provider.execution_context.provider_binding
        binding_match = $probeResult.provider.execution_context.provider_binding_match
        resolution_path = 'frozen execution-context artifact → bound profile row owner fingerprint → ProviderProfileService.list_profiles(exact owner scope) → frozen binding and safe fingerprint equality; read-only transaction, no global owner scan, no master-key load/create, no credential decrypt'
        fingerprint_contract = $probeResult.provider.fingerprint_contract
        profiles = $probeResult.provider.profiles
        provider_generation_calls = 0
        provider_http_count = 0
        db_writes = 0
        credential_plaintext_emitted = $false
        authorization_emitted = $false
    }
} catch {
    if (-not $failureCategory) {
        $message = $_.Exception.Message
        if ($message -match '^[A-Za-z0-9_:/\\.\-]+$') { $failureCategory = $message }
        else { $failureCategory = 'UNEXPECTED_' + $_.Exception.GetType().Name }
    }
} finally {
    if ($ephemeralName -and $dockerExe) {
        try {
            $cleanup = Invoke-Native $dockerExe @('rm','-f',$ephemeralName)
            if ($cleanup.ExitCode -eq 0) { $cleanupResult = 'REMOVED_EXACT_EPHEMERAL_CONTAINER' }
            else { $cleanupResult = 'EXACT_EPHEMERAL_CONTAINER_CLEANUP_FAILED' }
        } catch { $cleanupResult = 'EXACT_EPHEMERAL_CONTAINER_CLEANUP_FAILED' }
    }
}

$probeExecutionArtifact = [ordered]@{
    artifact_version = 'v2_1r_d4r_a6_operator.probe_execution.v1'
    status = if ($failureCategory) { 'FAILED' } elseif ($probeResult -and (Get-SafeProperty $probeResult 'status').value -eq 'PASS' -and $probeRun.ExitCode -eq 0) { 'PASS' } elseif ($probeRun) { 'FAILED' } else { 'NOT_STARTED' }
    compose_run = @{
        exit_code = if ($probeRun) { $probeRun.ExitCode } else { $null }
        timed_out = if ($probeRun) { $probeRun.TimedOut } else { $false }
        stdout_safe = $probeStdoutSafe
        stderr_safe = $probeStderrSafe
        start_time_utc = if ($probeRun) { $probeRun.StartedAtUtc } elseif ($probeStart) { $probeStart.ToString('o') } else { $null }
        end_time_utc = if ($probeRun) { $probeRun.EndedAtUtc } elseif ($probeEnd) { $probeEnd.ToString('o') } else { $null }
        duration_ms = $probeDurationMs
        stdout_present = -not [string]::IsNullOrWhiteSpace($probeStdoutSafe)
        stderr_present = -not [string]::IsNullOrWhiteSpace($probeStderrSafe)
        invocation_exception_type = $probeInvocationExceptionType
        invocation_failure = $probeInvocationFailure
    }
    ephemeral_runtime = @{
        container_name = $ephemeralName
        compose_service = 'api'
        image_reference = $apiImage
        image_id = $imageId
        image_inspect_completed = [bool]$imageId
        source_mount = @{ host_path=(Join-Path $repoRoot 'apps\api'); container_path='/app/apps/api'; read_only=$sourceMountVerified }
        current_checkout_runtime_identity = $currentCheckoutStatus
        cleanup_status = $cleanupResult
    }
    failure_category = $failureCategory
    failed_stage = $probeFailureStage
    provider_generation_calls = 0
    provider_http_count = 0
    semantic_verifier_calls = 0
    db_writes = 0
    holdout_access = 0
    diagnostics_redacted = $true
}
$probeStageArtifact = [ordered]@{
    artifact_version = 'v2_1r_d4r_a6_operator.probe_stages.v1'
    status = if ($hostProbeValidationFailure) { 'FAILED_VALIDATION' } elseif ($probeResult -and (Get-SafeProperty $probeResult 'status').value -eq 'PASS' -and -not $failureCategory) { 'PASS' } elseif ($probeResult) { 'FAILED_ENVELOPE' } else { 'NO_ENVELOPE' }
    stages = $probeMarkers
    first_failed_stage = $probeFailureStage
    import_results = $probeImports
    current_checkout_identity = (Get-SafeProperty $probeResult 'current_checkout_identity')
    python_identity = Get-ProbePythonIdentity $probeResult $probeMarkers
    database_diagnostics = (Get-SafeProperty $probeResult 'database')
    provider_diagnostics = (Get-SafeProperty $probeResult 'provider')
    host_validation_failure = $hostProbeValidationFailure
    safe_failure_envelope = if ((Get-SafeProperty $probeResult 'status').value -eq 'FAILED') { $probeResult } else { $null }
    classification = $failureCategory
    diagnostic_safety = @{
        stdout_and_stderr_redacted_before_persistence = $true
        raw_environment_dump = $false
        raw_dsn = $false
        credential_plaintext = $false
        authorization_header = $false
        raw_provider_response = $false
    }
    provider_generation_calls = 0
    provider_http_count = 0
    semantic_verifier_calls = 0
    db_writes = 0
    holdout_access = 0
}

if (-not $dockerHostAttestation) {
    $dockerHostAttestation = [ordered]@{
        artifact_version = 'v2_1r_d4r_a6_operator.docker_host.v1'
        produced_by = 'operator_host_powershell'
        host_identity = $hostIdentity
        repository_path = $repoRoot
        timestamp_utc = [DateTime]::UtcNow.ToString('o')
        status = 'BLOCKED'
        failure_category = $failureCategory
        docker_prechecks = $dockerPrechecks
        provider_generation_calls = 0; provider_http_count = 0; semantic_verifier_calls = 0; db_writes = 0; real_acquisition = 0; holdout_access = 0
    }
}
$dockerHostAttestation['ephemeral_container'] = @{
    name=$ephemeralName; compose_service='api'; image_reference=$apiImage; image_id=$imageId
    compose_run_exit_code=if($probeRun){$probeRun.ExitCode}else{$null}
    source_mount_current_checkout_read_only=$sourceMountVerified
    removed_by_operator_script=($cleanupResult -eq 'REMOVED_EXACT_EPHEMERAL_CONTAINER')
}
$dockerHostAttestation['ephemeral_cleanup'] = $cleanupResult
if (-not $runtimeAttestation) {
    $runtimeAttestation = [ordered]@{
        artifact_version='v2_1r_d4r_a6_operator.canonical_runtime.v1'; status=$runtimeStatus; failure_category=$failureCategory; failed_stage=$probeFailureStage
        current_checkout_identity=$currentCheckoutStatus; image_reference=$apiImage; image_id=$imageId; ephemeral_container_name=$ephemeralName
        import_results=$probeImports; stage_markers=$probeMarkers; provider_generation_calls=0; provider_http_count=0; semantic_verifier_calls=0; db_writes=0; holdout_access=0
    }
}
if (-not $providerAttestation) {
    $providerProperty = Get-SafeProperty $probeResult 'provider'
    $providerDiagnostic = if ($providerProperty.state -eq 'PRESENT') { $providerProperty.value } else { $null }
    $statusProperty = Get-SafeProperty $providerDiagnostic 'status'
    $providerFailureCategoryProperty = Get-SafeProperty $providerDiagnostic 'failure_category'
    $profilesProperty = Get-SafeProperty $providerDiagnostic 'profiles'
    $contractObservation = Get-ProviderFingerprintContractObservation $probeResult
    $contractState = $contractObservation.state
    $providerAttestation = [ordered]@{
        artifact_version='v2_1r_d4r_a6_operator.provider_readiness.v1';
        status=if ($providerStatus -eq 'FAILED') { 'FAILED' } elseif ($statusProperty.state -eq 'PRESENT') { $statusProperty.value } else { $providerStatus };
        resolution_path='read-only ProviderProfileService path; see stage diagnostics';
        failure_category=if ($providerFailureCategoryProperty.state -eq 'PRESENT') { $providerFailureCategoryProperty.value } else { $failureCategory };
        fingerprint_contract=if ($contractObservation.state -eq 'PRESENT') { $contractObservation.value } else { $null };
        fingerprint_contract_state=$contractState;
        fingerprint_contract_source=$contractObservation.source_path;
        profiles=if ($profilesProperty.state -eq 'PRESENT') { $profilesProperty.value } else { @() };
        provider_generation_calls=0; provider_http_count=0; semantic_verifier_calls=0; db_writes=0; credential_plaintext_emitted=$false; authorization_emitted=$false
    }
}

Write-JsonFile (Join-Path $outputDirectory 'docker_host_attestation.json') $dockerHostAttestation
Write-JsonFile (Join-Path $outputDirectory 'operator_probe_execution.json') $probeExecutionArtifact
Write-JsonFile (Join-Path $outputDirectory 'operator_probe_stage_report.json') $probeStageArtifact
Write-JsonFile (Join-Path $outputDirectory 'canonical_runtime_attestation.json') $runtimeAttestation
Write-JsonFile (Join-Path $outputDirectory 'provider_readiness_attestation.json') $providerAttestation

$manifestFiles = [ordered]@{}
foreach ($name in @('docker_host_attestation.json','operator_probe_execution.json','operator_probe_stage_report.json','canonical_runtime_attestation.json','provider_readiness_attestation.json')) {
    $manifestFiles[$name] = Get-Sha256 (Join-Path $outputDirectory $name)
}
$manifestFiles['scripts/operator_verify_canonical_runtime.ps1'] = Get-Sha256 $PSCommandPath
$manifestPayload = [ordered]@{ artifact_version='v2_1r_d4r_a6_operator.hash_manifest.v1'; algorithm='SHA-256'; files=$manifestFiles }
$manifestCanonical = ConvertTo-Json -InputObject $manifestPayload -Depth 20 -Compress
$sha = [System.Security.Cryptography.SHA256]::Create()
$manifestHash = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($manifestCanonical))
$sha.Dispose()
$manifestFilesHash = ([BitConverter]::ToString($manifestHash)).Replace('-','').ToLowerInvariant()
$hashManifest = [ordered]@{ artifact_version=$manifestPayload.artifact_version; algorithm='SHA-256'; files=$manifestFiles; manifest_sha256=$manifestFilesHash; self_hash_excluded=$true }
Write-JsonFile (Join-Path $outputDirectory 'hash_manifest.json') $hashManifest

if ($failureCategory -or $runtimeStatus -ne 'PASS' -or $providerStatus -ne 'RESOLVED' -or ($ephemeralName -and $cleanupResult -ne 'REMOVED_EXACT_EPHEMERAL_CONTAINER')) {
    Write-Error "Canonical runtime attestation blocked: $failureCategory; see $outputDirectory"
    exit 1
}
Write-Output "Canonical runtime attestation complete: $outputDirectory"
exit 0
