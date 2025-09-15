@echo off
REM install-dependencies.bat

REM Activate conda environment
CALL "%USERPROFILE%\anaconda3\Scripts\activate.bat" pygrex-exp-grs

REM Install editable package
pip install -e .

REM Verify if pygrex is importable
python -c "import pygrex; print('pygrex imported successfully')"