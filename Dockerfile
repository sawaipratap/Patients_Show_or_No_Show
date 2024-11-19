FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libhdf5-dev && pip install h5py

RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    python3-dev \
    libatlas-base-dev \
    --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "st_app.py", "--server.port=8501", "--server.address=0.0.0.0"]