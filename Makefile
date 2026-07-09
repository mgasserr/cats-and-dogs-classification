.PHONY: setup run

setup:
	@echo "=== 1. Creating Virtual Environment ==="
	cd backend && python -m venv venv
	@echo "=== 2. Upgrading pip ==="
	cd backend && venv\Scripts\python -m pip install --upgrade pip
	@echo "=== 3. Installing Dependencies ==="
	cd backend && venv\Scripts\pip install -r requirements.txt
	@echo "=== 4. Getting the Model (Downloading dataset & training) ==="
	cd backend && venv\Scripts\python train.py
	@echo "=== Setup Complete! You can now type 'make run' ==="

run:
	@echo "=== Opening Website in Default Browser ==="
	start "" "frontend\index.html"
	@echo "=== Starting FastAPI Backend Server ==="
	cd backend && venv\Scripts\uvicorn main:app --reload --port 8000