# Create quantumsentiment PostgreSQL database and user (Windows)
# Requires: PostgreSQL service running, admin access to pg_hba.conf for first-time setup
#
# Usage:
#   .\scripts\setup_postgres_db.ps1 -PostgresPassword "your-postgres-superuser-password"
# Or set POSTGRES_SUPERUSER_PASSWORD in the environment.

param(
    [string]$PostgresPassword = $env:POSTGRES_SUPERUSER_PASSWORD,
    [string]$DbName = "quantumsentiment",
    [string]$DbUser = "quantumsentiment",
    [int]$Port = 5432,
    [string]$Host = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$pgBin = "C:\Program Files\PostgreSQL\18\bin"
$psql = Join-Path $pgBin "psql.exe"
$createdb = Join-Path $pgBin "createdb.exe"

if (-not (Test-Path $psql)) {
    throw "psql not found at $psql. Adjust `$pgBin for your PostgreSQL install."
}
if (-not $PostgresPassword) {
    $secure = Read-Host "PostgreSQL superuser (postgres) password" -AsSecureString
    $PostgresPassword = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    )
}

function New-DbPassword {
    -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 24 | ForEach-Object { [char]$_ })
}

$DbPassword = New-DbPassword
$env:PGPASSWORD = $PostgresPassword

& $psql -U postgres -h $Host -p $Port -d postgres -v ON_ERROR_STOP=1 -c @"
DO `$`$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '$DbUser') THEN
    CREATE ROLE $DbUser WITH LOGIN PASSWORD '$DbPassword';
  ELSE
    ALTER ROLE $DbUser WITH LOGIN PASSWORD '$DbPassword';
  END IF;
END
`$`$;
"@

$dbExists = & $psql -U postgres -h $Host -p $Port -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname = '$DbName';"
if ($dbExists -ne "1") {
    & $createdb -U postgres -h $Host -p $Port -O $DbUser $DbName
}

$env:PGPASSWORD = $DbPassword
& $psql -U $DbUser -h $Host -p $Port -d $DbName -v ON_ERROR_STOP=1 -c @"
GRANT ALL ON SCHEMA public TO $DbUser;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO $DbUser;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO $DbUser;
"@

$databaseUrl = "postgresql://${DbUser}:${DbPassword}@${Host}:${Port}/${DbName}"
$envFile = Join-Path (Split-Path $PSScriptRoot -Parent) ".env"
if (Test-Path $envFile) {
    $lines = Get-Content $envFile
    $updated = $false
    $lines = $lines | ForEach-Object {
        if ($_ -match '^DATABASE_URL=') {
            $updated = $true
            "DATABASE_URL=$databaseUrl"
        } else { $_ }
    }
    if (-not $updated) { $lines += "DATABASE_URL=$databaseUrl" }
    Set-Content -Path $envFile -Value $lines
}

Write-Host "Database '$DbName' ready."
Write-Host "Updated $envFile with DATABASE_URL."
Write-Host "Run table init: python -c (see README database test)"

$env:PGPASSWORD = $null
