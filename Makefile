.PHONY: help install data features train optimize explain monitor pipeline api dashboard test lint docker-build docker-up docker-down clean

export PYTHONPATH := $(CURDIR)

help:
	@echo "Demand Forecasting & Inventory Optimization Platform"
	@echo ""
	@echo "  make install     install dependencies"
	@echo "  make pipeline    run the full pipeline (data -> train -> optimize -> explain -> monitor)"
	@echo "  make api         serve the FastAPI app on :8000"
	@echo "  make dashboard   serve the Streamlit dashboard on :8501"
	@echo "  make mlflow      serve the MLflow UI on :5000"
	@echo "  make test        run the test suite"
	@echo "  make docker-up   build and run the whole stack via docker compose"
	@echo "  make clean       remove generated data and artifacts"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

data:
	python -m src.data.generate_data

features: data
	python -m src.features.build_features

train:
	python -m src.models.train

optimize:
	python -m scripts.run_optimization

explain:
	python -m src.models.explain

monitor:
	-python -m scripts.run_monitoring

# Full run. `monitor` exits non-zero when a retrain trigger fires, so it is
# prefixed with `-` to keep that from aborting the make target.
pipeline: features train optimize explain monitor
	@echo ""
	@echo "Pipeline complete. Artifacts in ./artifacts"

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run dashboard/app.py --server.port=8501

mlflow:
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

test:
	pytest tests/ -q

lint:
	ruff check src/ scripts/ tests/ dashboard/ --output-format=concise

docker-build:
	docker build -t demand-forecast-platform:latest .

docker-up:
	docker compose up --build -d
	@echo "API       -> http://localhost:8000/docs"
	@echo "Dashboard -> http://localhost:8501"
	@echo "MLflow    -> http://localhost:5000"

docker-down:
	docker compose down -v

clean:
	rm -rf data/raw/*.parquet data/processed/*.parquet
	rm -rf artifacts/*.parquet artifacts/*.csv artifacts/*.json artifacts/*.png
	rm -rf mlruns mlflow.db .pytest_cache reports
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "cleaned"
