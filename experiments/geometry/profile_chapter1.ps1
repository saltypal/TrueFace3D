param(
    [string]$Output = 'experiments/geometry/runs/chapter1-flops',
    [string]$Distro = 'Ubuntu-26.04',
    [string]$Python = '/home/Satya/.venvs/trueface3d-vit/bin/python'
)

$ErrorActionPreference = 'Stop'
$root = 'experiments/geometry/runs'
if (!(Test-Path 'pyproject.toml')) {
    throw 'Run this script from the TrueFace3D repository root.'
}

function Get-OnePath($items) {
    $resolved = @($items)
    if ($resolved.Count -ne 1) { throw "Expected one profile input, found $($resolved.Count)." }
    $resolved[0].FullName
}

function ConvertTo-WslPath([string]$Path) {
    '/mnt/' + $Path.Substring(0, 1).ToLower() + '/' + $Path.Substring(3).Replace('\', '/')
}

$image = Get-OnePath (Get-ChildItem Data/NoWDataset -Recurse -Filter IMG_0041.jpg |
    Where-Object FullName -Match 'FaMoS_180424_03335_TA')
$pixel = Get-OnePath (Get-ChildItem "$root/chapter1-pixel-original-03/cases" -Recurse -Directory -Filter single |
    Where-Object { $_.Parent.Name -eq 'preprocessed' })
$inputs = [ordered]@{}
foreach ($model in @('vits', 'vitb')) {
    $inputs[$model] = Get-OnePath (Get-ChildItem "$root/chapter1-original-01/cases" -Recurse -Filter priors.npz |
        Where-Object FullName -Match "[/\\]$model[/\\]")
}
$inputs['deca'] = $image
$inputs['mica'] = $image
$inputs['uv_map'] = $pixel
$inputs['normals'] = $pixel

New-Item -Path $Output -ItemType Directory -Force | Out-Null
$directory = (Resolve-Path $Output).Path
foreach ($model in $inputs.Keys) {
    $receipt = Join-Path $directory "$model.json"
    if (Test-Path $receipt) { continue }
    Write-Output "Profiling $model"
    wsl.exe -d $Distro --exec env PYTHONPATH=. TORCHDYNAMO_DISABLE=1 $Python -m src.geometry.profile_flops $model (ConvertTo-WslPath $inputs[$model]) (ConvertTo-WslPath $receipt)
    if ($LASTEXITCODE -ne 0) { throw "$model profile failed with exit code $LASTEXITCODE." }
}
