if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Output "uv is not installed. Installing now..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    Write-Output "uv has been installed successfully. Restart your terminal to ensure the changes take effect."
    exit
} else {
    Write-Output "uv is already installed.  Launching dashboard..."
    uv run --no-dev dashboard -H 0.0.0.0 --no-debug
}
