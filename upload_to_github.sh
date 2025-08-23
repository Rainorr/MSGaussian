#!/bin/bash

echo "🚀 Uploading GaussianLSS MindSpore to GitHub"
echo "============================================"

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d "gaussianlss_ms" ]; then
    echo "❌ Error: Please run this script from the GaussianLSS_MindSpore directory"
    exit 1
fi

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
    git branch -m main
fi

# Check if remote exists
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "🔗 Adding GitHub remote..."
    read -p "Enter your GitHub username: " username
    git remote add origin https://github.com/$username/mindspore_gaussian.git
fi

echo "📝 Adding files to git..."
git add .

echo "💾 Committing changes..."
git commit -m "Initial commit: GaussianLSS PyTorch to MindSpore migration

- Complete project structure with modular architecture
- Data pipeline with MindSpore Dataset API integration  
- Model architecture including GaussianLSS, backbones, and Gaussian renderer
- Comprehensive loss functions (Focal Loss, Smooth L1 Loss)
- Evaluation metrics system
- Training pipeline with YAML configuration
- Documentation and migration reports
- 75% core functionality complete

Key features:
✅ Data processing (NuScenesDataset, transforms, data module)
✅ Model architecture (GaussianLSS, EfficientNet, ResNet backbones)
✅ Gaussian renderer with pure MindSpore implementation
✅ Loss functions and metrics
✅ Training infrastructure
✅ Configuration system
✅ Comprehensive documentation"

echo "🚀 Pushing to GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo "✅ Successfully uploaded to GitHub!"
    echo "🔗 Repository URL: https://github.com/$username/mindspore_gaussian"
else
    echo "❌ Upload failed. Please check your GitHub credentials and repository settings."
    echo ""
    echo "Manual upload instructions:"
    echo "1. Create the repository on GitHub: https://github.com/new"
    echo "2. Run: git remote set-url origin https://github.com/YOUR_USERNAME/mindspore_gaussian.git"
    echo "3. Run: git push -u origin main"
fi