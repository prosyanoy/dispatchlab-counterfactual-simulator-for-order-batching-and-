FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY dispatchlab ./dispatchlab
COPY pyproject.toml README.md ./
ENV DISPATCHLAB_DATA_DIR=/app/data DISPATCHLAB_OUTPUT_DIR=/app/outputs MPLBACKEND=Agg PYTHONUNBUFFERED=1
VOLUME ["/app/outputs"]
# Default: generate data, train, simulate, run OPE and write outputs/report.md (+ charts)
CMD ["python", "-m", "dispatchlab", "all"]
