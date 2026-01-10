FROM python:3.11-slim-bookworm
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1 torchvision==0.20.1
    
COPY requirements.runtime.txt /app/requirements.runtime.txt
RUN pip install --no-cache-dir -r requirements.runtime.txt

COPY src /app/src
COPY src/fer/train/*.pt /app/src/fer/train/

ENV PYTHONUNBUFFERED=1 HF_HOME=/app/.hf_cache
EXPOSE 8000
CMD ["uvicorn", "src.backen.main:app", "--host", "0.0.0.0", "--port", "8000"]
