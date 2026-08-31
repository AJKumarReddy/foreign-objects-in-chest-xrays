<#
.SYNOPSIS
  Windows task runner - the Makefile targets, for PowerShell.
.EXAMPLE
  ./tasks.ps1 dev-api
#>
param([Parameter(Position = 0)][string]$Task = "help")

switch ($Task) {
    "install"     { pip install -e backend }
    "install-dev" { pip install -e "backend[dev]"; Push-Location frontend; npm install; Pop-Location }
    "dev-api"     { python -m cxr.cli serve --reload }
    "dev-web"     { Push-Location frontend; npm run dev; Pop-Location }
    "test"        { Push-Location backend; python -m pytest -q; Pop-Location
                    Push-Location frontend; npm test; Pop-Location }
    "lint"        { ruff check backend; Push-Location frontend; npm run lint; Pop-Location }
    "build"       { Push-Location frontend; npm run build; Pop-Location }
    "samples"     { python scripts/generate_demo_samples.py }
    "docker"      { docker compose up --build }
    default       { Write-Host "tasks: install install-dev dev-api dev-web test lint build samples docker" }
}
