function ConvertTo-SafeHostDiagnosticText([AllowNull()][string]$Text) {
    if ($null -eq $Text) { return $null }
    $safe = $Text -replace '(?i)(://[^:/\s]+:)[^@/\s]+(@)', '$1[REDACTED]$2'
    $safe = $safe -replace '(?i)(api[_-]?key|authorization|password|secret|token)(\s*["'']?\s*[:=]\s*["'']?)[^,\s"'';]+', '$1$2[REDACTED]'
    if ($safe.Length -gt 12000) { $safe = $safe.Substring(0, 12000) + "`n[TRUNCATED]" }
    return $safe
}

function ConvertTo-SafeHostDiagnosticArguments([string[]]$Arguments) {
    $safeArguments = [System.Collections.Generic.List[string]]::new()
    $redactNextEnvironment = $false
    foreach ($argument in $Arguments) {
        if ($redactNextEnvironment) {
            $safeArguments.Add((ConvertTo-SafeHostDiagnosticText $argument))
            $redactNextEnvironment = $false
            continue
        }
        if ($argument -eq '--env') {
            $safeArguments.Add($argument)
            $redactNextEnvironment = $true
            continue
        }
        $safeArguments.Add((ConvertTo-SafeHostDiagnosticText $argument))
    }
    return $safeArguments.ToArray()
}

function New-HostCommandFailureDiagnostic {
    param(
        [Parameter(Mandatory)][string]$RunId,
        [Parameter(Mandatory)][string]$FailureGate,
        [Parameter(Mandatory)][string]$Executable,
        [Parameter(Mandatory)][string[]]$Arguments,
        [AllowNull()][Nullable[int]]$ExitCode,
        [AllowNull()][string]$InvocationException,
        [AllowNull()][string]$Stdout,
        [AllowNull()][string]$Stderr,
        [Parameter(Mandatory)][string]$WorkingDirectory
    )
    $safeArguments = @(ConvertTo-SafeHostDiagnosticArguments $Arguments)
    $safeCommand = @($Executable) + $safeArguments
    $containerWorkdir = $null
    $workdirIndex = [Array]::IndexOf($Arguments, '--workdir')
    if ($workdirIndex -ge 0 -and $workdirIndex + 1 -lt $Arguments.Count) {
        $containerWorkdir = $Arguments[$workdirIndex + 1]
    }
    $environment = @(
        for ($index = 0; $index -lt $Arguments.Count; $index++) {
            if ($Arguments[$index] -eq '--env' -and $index + 1 -lt $Arguments.Count) {
                ConvertTo-SafeHostDiagnosticText $Arguments[$index + 1]
                $index++
            }
        }
    )
    $safeException = if ([string]::IsNullOrEmpty($InvocationException)) {
        $null
    } else {
        ConvertTo-SafeHostDiagnosticText $InvocationException
    }
    return [ordered]@{
        diagnostic_version = 'prospective-acquisition-host-command-failure.v1'
        run_id = $RunId
        failure_gate = $FailureGate
        command = [ordered]@{
            executable = $Executable
            arguments = $safeArguments
            command_line_length = (($safeCommand | ForEach-Object { [string]$_ }) -join ' ').Length
            host_working_directory = $WorkingDirectory
            container_working_directory = $containerWorkdir
            environment = $environment
        }
        exit_code = $ExitCode
        invocation_exception = $safeException
        stdout = ConvertTo-SafeHostDiagnosticText $Stdout
        stderr = ConvertTo-SafeHostDiagnosticText $Stderr
    }
}

function Write-HostCommandFailureArtifact([string]$RepositoryRoot, [object]$Diagnostic) {
    $timestamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
    $runDirectory = Join-Path $RepositoryRoot "artifacts\v2_1r_d4r_execution_lane_host\run_${timestamp}_$($Diagnostic.run_id)"
    [void](New-Item -ItemType Directory -Path $runDirectory -Force)
    $path = Join-Path $runDirectory 'command_failure.json'
    $json = $Diagnostic | ConvertTo-Json -Depth 12
    [System.IO.File]::WriteAllText($path, $json, [System.Text.UTF8Encoding]::new($false))
    return $path
}

function Get-SafePytestSummary([AllowNull()][string]$Text) {
    if ([string]::IsNullOrEmpty($Text)) { return $null }
    $safeLines = [System.Collections.Generic.List[string]]::new()
    foreach ($line in ($Text -split "`r?`n")) {
        if ($line -match '^\s*FAILED\s+(\S+)') {
            $safeLines.Add("FAILED $($Matches[1])")
        } elseif ($line -match '^\s*=+\s*(\d+\s+(?:failed|passed|error|errors|skipped|xfailed|xpassed).*)\s*=+\s*$') {
            $safeLines.Add($Matches[1])
        } elseif ($line -match '^\s*(?:E\s+)?([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Failure))(?::.*)?$') {
            $safeLines.Add("exception_type=$($Matches[1])")
        } elseif ($line -match '^\s*collected\s+\d+\s+items?\s*$') {
            $safeLines.Add($line.Trim())
        }
    }
    if ($safeLines.Count -eq 0) { return '[No safe pytest summary lines were parseable]' }
    return (($safeLines | Select-Object -Unique) -join "`n")
}

function New-TargetedHostTestFailureDiagnostic {
    param(
        [Parameter(Mandatory)][string]$RunId,
        [Parameter(Mandatory)][string]$Executable,
        [Parameter(Mandatory)][string[]]$Arguments,
        [Parameter(Mandatory)][int]$ExitCode,
        [AllowNull()][string]$Stdout,
        [AllowNull()][string]$Stderr,
        [Parameter(Mandatory)][string]$WorkingDirectory
    )
    $nodeId = $null
    foreach ($line in @($Stdout, $Stderr) -split "`r?`n") {
        if ($line -match '^\s*FAILED\s+(\S+::\S+)') {
            $nodeId = $Matches[1]
            break
        }
    }
    $databaseFailure = Get-SafeDatabaseFailureMetadata -Text @($Stdout, $Stderr)
    return [ordered]@{
        diagnostic_version = 'prospective-acquisition-targeted-test-failure.v1'
        run_id = $RunId
        gate = 'TARGETED_HOST_INTEGRATION_TESTS'
        command = [ordered]@{
            executable = $Executable
            arguments = $Arguments
            working_directory = $WorkingDirectory
            environment_names = @(
                'RUN_PROSPECTIVE_LANE_HOST_ATTESTATION'
                'PROSPECTIVE_LANE_DISPOSABLE_DB'
                'TEST_DATABASE_URL'
                'PYTHONDONTWRITEBYTECODE'
            )
        }
        exit_code = $ExitCode
        failed_test_nodeid = $nodeId
        dbapi_exception_type = $databaseFailure.dbapi_exception_type
        sqlstate = $databaseFailure.sqlstate
        root_cause_category = $databaseFailure.root_cause_category
        safe_stdout = Get-SafePytestSummary $Stdout
        safe_stderr = Get-SafePytestSummary $Stderr
        raw_output_preserved = $false
        task_text_preserved = $false
    }
}

function Get-SafeDatabaseFailureMetadata {
    param([AllowNull()][string[]]$Text)
    $combined = (@($Text) -join "`n")
    $dbapiType = $null
    $dbapiMatch = [regex]::Match(
        $combined,
        '(?i)\b(psycopg(?:2)?(?:\.errors)?\.[A-Za-z_][A-Za-z0-9_.]*)'
    )
    if ($dbapiMatch.Success) { $dbapiType = $dbapiMatch.Groups[1].Value }

    $sqlstate = $null
    $stateMatch = [regex]::Match(
        $combined,
        '(?i)(?:sqlstate|sql_state|pgcode)\s*[=: ]\s*["'']?([0-9A-Z]{5})'
    )
    if ($stateMatch.Success) { $sqlstate = $stateMatch.Groups[1].Value.ToUpperInvariant() }

    $lower = $combined.ToLowerInvariant()
    $category = 'UNKNOWN_DATABASE_ERROR'
    if ($lower -match 'could not translate host name|name or service not known|getaddrinfo failed|nodename nor servname') {
        $category = 'HOST_NOT_RESOLVED'
    } elseif ($lower -match 'connection refused') {
        $category = 'CONNECTION_REFUSED'
    } elseif ($lower -match 'password authentication failed|authentication failed|no pg_hba.conf entry' -or $sqlstate -eq '28P01') {
        $category = 'AUTHENTICATION_FAILED'
    } elseif ($lower -match 'database .* does not exist' -or $sqlstate -eq '3D000') {
        $category = 'DATABASE_NOT_FOUND'
    } elseif ($lower -match 'invalid input value for enum' -or $sqlstate -eq '22P02') {
        $category = 'INVALID_ENUM_VALUE'
    } elseif ($dbapiType -match '\.UndefinedColumn$' -or $sqlstate -eq '42703') {
        $category = 'UNDEFINED_COLUMN'
    } elseif ($dbapiType -match '\.UndefinedTable$' -or $sqlstate -eq '42P01') {
        $category = 'MISSING_SCHEMA'
    } elseif ($lower -match 'timeout expired|connection timed out|connect timeout') {
        $category = 'CONNECTION_TIMEOUT'
    }

    return [ordered]@{
        dbapi_exception_type = $dbapiType
        sqlstate = $sqlstate
        root_cause_category = $category
    }
}

function Write-TargetedHostTestFailureArtifact([string]$RepositoryRoot, [object]$Diagnostic) {
    $timestamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
    $runDirectory = Join-Path $RepositoryRoot "artifacts\v2_1r_d4r_execution_lane_host\run_${timestamp}_$($Diagnostic.run_id)"
    [void](New-Item -ItemType Directory -Path $runDirectory -Force)
    $path = Join-Path $runDirectory 'targeted_test_failure.json'
    $json = $Diagnostic | ConvertTo-Json -Depth 10
    [System.IO.File]::WriteAllText($path, $json, [System.Text.UTF8Encoding]::new($false))
    return $path
}

function Write-HostPostgresReadinessFailureArtifact {
    param(
        [Parameter(Mandatory)][string]$RepositoryRoot,
        [Parameter(Mandatory)][string]$RunId,
        [Parameter(Mandatory)][object]$Readiness
    )
    $timestamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
    $runDirectory = Join-Path $RepositoryRoot "artifacts\v2_1r_d4r_execution_lane_host\run_${timestamp}_$RunId"
    [void](New-Item -ItemType Directory -Path $runDirectory -Force)
    $path = Join-Path $runDirectory 'host_postgres_readiness_failure.json'
    $safe = [ordered]@{
        diagnostic_version = 'prospective-acquisition-host-postgres-readiness.v1'
        run_id = $RunId
        gate = 'HOST_POSTGRES_READINESS'
        status = [string]$Readiness.status
        host = [string]$Readiness.host
        port = $Readiness.port
        database = [string]$Readiness.database
        attempt_count = $Readiness.attempt_count
        elapsed_ms = $Readiness.elapsed_ms
        dbapi_exception_type = $Readiness.dbapi_exception_type
        sqlstate = $Readiness.sqlstate
        root_cause_category = [string]$Readiness.root_cause_category
        credentials_redacted = $true
        connection_url_preserved = $false
        raw_exception_preserved = $false
    }
    [System.IO.File]::WriteAllText(
        $path,
        ($safe | ConvertTo-Json -Depth 6),
        [System.Text.UTF8Encoding]::new($false)
    )
    return $path
}
