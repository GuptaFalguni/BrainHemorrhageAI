# Package deployable model weights for GitHub Release (not committed to git).
# Run from repo root in PowerShell:
#   .\scripts\package_weights.ps1
# Then:
#   gh release create v1.0.0 .\dist\model-weights-v1.tar.gz --title "v1.0.0" --notes "Deployable MONAI + nnU-Net checkpoints"

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$Required = @(
    "checkpoints\experiment_best30h\best_model.pt",
    "checkpoints\nnunet_dataset501_fold0\checkpoint_best.pth",
    "checkpoints\nnunet_dataset501_fold0\nnUNetPlans.json",
    "checkpoints\nnunet_dataset501_fold0\dataset.json"
)

foreach ($rel in $Required) {
    $path = Join-Path $Root $rel
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required file: $rel"
    }
}

$Dist = Join-Path $Root "dist"
New-Item -ItemType Directory -Force -Path $Dist | Out-Null
$Out = Join-Path $Dist "model-weights-v1.tar.gz"
if (Test-Path $Out) { Remove-Item -Force $Out }

# tar includes the checkpoints/ prefix so entrypoint can extract into /app
tar -czf $Out `
    "checkpoints/experiment_best30h/best_model.pt" `
    "checkpoints/nnunet_dataset501_fold0/checkpoint_best.pth" `
    "checkpoints/nnunet_dataset501_fold0/nnUNetPlans.json" `
    "checkpoints/nnunet_dataset501_fold0/dataset.json"

$sizeMb = [math]::Round((Get-Item $Out).Length / 1MB, 1)
Write-Host "Created $Out ($sizeMb MB)"
Write-Host "Publish with:"
Write-Host "  gh release create v1.0.0 `"$Out`" --title `"v1.0.0`" --notes `"Deployable MONAI + nnU-Net checkpoints for Docker.`""
