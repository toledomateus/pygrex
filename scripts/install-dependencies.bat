@echo off
REM install-dependencies.bat

REM Activate conda environment
CALL "%USERPROFILE%\anaconda3\Scripts\activate.bat" recoxplainer_add-sliding-window

REM Install editable package
pip install -e .
