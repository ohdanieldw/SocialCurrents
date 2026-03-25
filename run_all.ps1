# Simple helper to install dependencies with Poetry and run the full pipeline.

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PipelineArgs
)

$ErrorActionPreference = "Stop"      

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Write-Host "[ERROR] '$Name' is required but not installed." -ForegroundColor Red
        Write-Host "Install Poetry from https://python-poetry.org/docs/#installation" -ForegroundColor Yellow    
        exit 1
    }
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

Require-Command -Name "poetry"       

Write-Host "[INFO] Installing project dependencies via Poetry..." -ForegroundColor Blue
poetry install

Write-Host "[INFO] Running multimodal data pipeline..." -ForegroundColor Blue
poetry run python run_pipeline.py @PipelineArgs

Write-Host "[INFO] Pipeline run completed." -ForegroundColor Green
