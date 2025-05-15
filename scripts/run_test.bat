@echo off
coverage run --source=recoxplainer --branch -m pytest test -v
coverage report -m
coverage html
