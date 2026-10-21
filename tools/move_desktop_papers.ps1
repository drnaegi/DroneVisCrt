[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Source,
    [string]$Destination = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $Source -PathType Container)) {
    throw "Source directory does not exist: $Source"
}
if (-not (Test-Path -LiteralPath $Destination -PathType Container)) {
    throw "Destination directory does not exist: $Destination"
}

$sourcePath = (Get-Item -LiteralPath $Source -Force).FullName.TrimEnd('\', '/')
$destinationPath = (Get-Item -LiteralPath $Destination -Force).FullName.TrimEnd('\', '/')
$sourcePrefix = $sourcePath + [System.IO.Path]::DirectorySeparatorChar

if ($sourcePath.Equals($destinationPath, [System.StringComparison]::OrdinalIgnoreCase) -or
    $destinationPath.StartsWith($sourcePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Destination must be outside the source directory.'
}

$items = @(Get-ChildItem -LiteralPath $sourcePath -Force)
$conflicts = @($items | Where-Object {
    Test-Path -LiteralPath (Join-Path -Path $destinationPath -ChildPath $_.Name)
})

if ($conflicts.Count -gt 0) {
    throw ('No files moved. Destination already contains: ' + ($conflicts.Name -join ', '))
}

$movedCount = 0
foreach ($item in $items) {
    $targetPath = Join-Path -Path $destinationPath -ChildPath $item.Name
    if (Test-Path -LiteralPath $targetPath) {
        throw "Stopped after moving $movedCount entries. Destination already contains: $targetPath"
    }

    Move-Item -LiteralPath $item.FullName -Destination $targetPath

    if ((Test-Path -LiteralPath $item.FullName) -or -not (Test-Path -LiteralPath $targetPath)) {
        throw "Move verification failed: $($item.FullName)"
    }

    $movedCount++
    Write-Output "Moved: $($item.Name)"
}

Write-Output "Moved $movedCount files or folders to: $destinationPath"
