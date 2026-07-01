@echo off
echo ========================================
echo SignBERT 50 Classes - Hugging Face Deployment
echo ========================================
echo.

cd /d "%~dp0"

echo [1/5] Initializing Git repository...
git init

echo.
echo [2/5] Adding remote repository...
git remote add origin https://huggingface.co/spaces/NovatrixSolutions/Sign_Language_Detection_SIGNBERT

echo.
echo [3/5] Adding all files...
git add .

echo.
echo [4/5] Committing files...
git commit -m "Deploy SignBERT 50 Classes Model"

echo.
echo [5/5] Pushing to Hugging Face...
echo When prompted, use your access token as password:
echo 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
echo.
git push -u origin main

echo.
echo ========================================
echo Deployment Complete!
echo ========================================
echo Your Space will be available at:
echo https://huggingface.co/spaces/NovatrixSolutions/Sign_Language_Detection_SIGNBERT
echo.
pause
