@echo off
REM install-dependencies.bat

REM Activate conda environment
CALL "%USERPROFILE%\anaconda3\Scripts\activate.bat" recoxplainer_add-sliding-window

REM Install editable package
pip install -e .

REM Verify if recoxplainer is importable
python -c "import recoxplainer; print('recoxplainer imported successfully')"