# Setup script for Windows PowerShell
# UAV Multi-Task Model Inference Project

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "UAV-MTM-Inference Setup" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# Check Python version
Write-Host "`nChecking Python version..." -ForegroundColor Yellow
$pythonVersion = python --version
Write-Host $pythonVersion -ForegroundColor Green

# Check if in correct directory
if (-not (Test-Path "requirements.txt")) {
    Write-Host "Error: requirements.txt not found. Are you in the project root?" -ForegroundColor Red
    exit 1
}

# Install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt

# Check PyTorch installation
Write-Host "`nVerifying PyTorch installation..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Create necessary directories
Write-Host "`nCreating project directories..." -ForegroundColor Yellow
$dirs = @(
    "data/raw",
    "data/processed",
    "data/pretrained",
    "experiments/accuracy_fitting",
    "experiments/training",
    "checkpoints",
    "logs",
    "results/figures",
    "results/tables"
)

foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created: $dir" -ForegroundColor Green
    }
}

Write-Host "`n" + "=" * 60 -ForegroundColor Cyan
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Download NYUv2 dataset: python scripts/data/0_download_nyuv2.py"
Write-Host "2. Verify environment: python check_env.py"
Write-Host "3. Start with accuracy fitting: python scripts/1_fit_accuracy.py --model mtan"

