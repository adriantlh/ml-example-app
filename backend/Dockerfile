FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies required for building wheels and running OpenCV/ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies directly
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "fastapi>=0.117.1,<0.118.0" \
        "uvicorn>=0.36.0,<0.37.0" \
        "python-multipart>=0.0.20,<0.0.21" \
        "pillow>=11.3.0,<12.0.0" \
        "onnxruntime>=1.22.1,<2.0.0" \
        "opencv-python>=4.12.0.88,<5.0.0.0"

# Copy application
COPY backend /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
