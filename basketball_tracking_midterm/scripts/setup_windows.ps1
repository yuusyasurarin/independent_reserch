$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$oldPythonPath = $env:PYTHONPATH
$oldPipCache = $env:PIP_CACHE_DIR
try {
    Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    $env:PIP_CACHE_DIR = Join-Path $projectRoot '.cache\pip'
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        & py -3.12 -m venv (Join-Path $projectRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 venv creation failed' }
    }
    & $pythonPath -m pip install 'pip==24.3.1' 'setuptools==75.6.0'
    if ($LASTEXITCODE -ne 0) { throw 'pip setup failed' }
    & $pythonPath -m pip install 'torch==2.5.1' 'torchvision==0.20.1' --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
    & $pythonPath -m pip install -r (Join-Path $projectRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Requirements installation failed' }
    & $pythonPath -m pip install --no-deps 'supervision==0.25.0'
    if ($LASTEXITCODE -ne 0) { throw 'ByteTrack dependency installation failed' }
    & $pythonPath -c "import torch, cv2, supervision; print(torch.__version__, torch.cuda.is_available(), cv2.__version__, hasattr(cv2, 'TrackerKCF_create'), supervision.__version__)"
    if ($LASTEXITCODE -ne 0) { throw 'Runtime check failed' }
} finally {
    $env:PYTHONPATH = $oldPythonPath
    $env:PIP_CACHE_DIR = $oldPipCache
}
