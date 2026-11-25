FROM python:3.12-slim
# LibreOffice converts Word/PowerPoint/Excel to PDF
RUN apt-get update && apt-get install -y --no-install-recommends \
      libreoffice-core libreoffice-writer libreoffice-impress libreoffice-calc fonts-dejavu \
      libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY rag ./rag
COPY ui ./ui
COPY eval ./eval
COPY deploy ./deploy
ENV DATA_DIR=/data
EXPOSE 8000 8501
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
