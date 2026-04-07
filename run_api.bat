@echo off
call .venv\Scripts\activate
uvicorn src.v1_api:app --reload --port 8000