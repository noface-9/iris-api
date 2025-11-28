# Dockerfile
FROM python:3.11-slim

# set workdir
WORKDIR /app

# system deps if any (add gcc for some libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy requirements first to leverage cache
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

# copy app code
COPY main.py /app/
COPY model.pkl /app/

# expose port
EXPOSE 8000

# use a userless non-root in real usage (skipped for brevity)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
