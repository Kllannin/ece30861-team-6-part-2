FROM python:3.10-slim
WORKDIR /app

# Install Phase 1 dependencies first
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install API dependencies
COPY api/requirements.txt ./api_requirements.txt
RUN pip install -r api_requirements.txt

# Copy Phase 1 code (metrics, run.py, etc.)
COPY src/ ./src/
COPY run.py .
COPY get_model_metrics.py .
COPY metric_caller.py .
COPY tasks.txt .

# Copy API code to /app directory (not subdirectory)
COPY api/main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]