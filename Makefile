.PHONY: install install-dev dev-api dev-web test lint build docker clean samples

install:            ## install the backend package
	pip install -e backend

install-dev:        ## backend + dev extras + front end deps
	pip install -e "backend[dev]"
	cd frontend && npm install

dev-api:            ## run the API with auto-reload on :8000
	python -m cxr.cli serve --reload

dev-web:            ## run the Vite dev server on :5173
	cd frontend && npm run dev

test:               ## backend + front end unit tests
	cd backend && python -m pytest -q
	cd frontend && npm test

lint:
	ruff check backend
	cd frontend && npm run lint

build:              ## production build of the SPA
	cd frontend && npm run build

samples:            ## regenerate the synthetic demo assets
	python scripts/generate_demo_samples.py

docker:
	docker compose up --build

clean:
	rm -rf backend/**/__pycache__ .pytest_cache .ruff_cache frontend/dist
