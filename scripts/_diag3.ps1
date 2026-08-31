$paths = @(
    "$env:APPDATA\Docker\settings-store.json",
    "$env:APPDATA\Docker\settings.json",
    "$env:APPDATA\Docker\settings-store.json.bak"
)
foreach ($p in $paths) {
    if (Test-Path $p) {
        Write-Host "=== $p ==="
        Get-Content $p -Raw
        Write-Host ""
    }
}
