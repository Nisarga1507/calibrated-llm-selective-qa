FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md api.py ./
COPY configs/ ./configs/
COPY src/ ./src/
COPY data/splits/ ./data/splits/
COPY results/tables/ ./results/tables/

RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
