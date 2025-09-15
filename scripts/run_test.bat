@echo off
coverage run --source=pygrex --branch -m pytest test -v
coverage report -m
coverage html
