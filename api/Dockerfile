FROM python:3.12-slim

WORKDIR /app

# Copier les requirements
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le modèle (maintenant accessible depuis le contexte parent)
COPY outputs/ /app/outputs/

# Copier le code de l'API
COPY api/app/ /app/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]