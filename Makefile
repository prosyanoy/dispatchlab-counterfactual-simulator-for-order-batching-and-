.PHONY: install test report serve docker demo
install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
test:
	python -m pytest -q
report:
	PYTHONUNBUFFERED=1 python -m dispatchlab all
serve:
	python -m dispatchlab serve
docker:
	docker compose run --rm --build report
demo:  ## generate data, train, simulate, run OPE and write outputs/report.md (+ charts); ~2.5 min
	PYTHONUNBUFFERED=1 python -m dispatchlab all
