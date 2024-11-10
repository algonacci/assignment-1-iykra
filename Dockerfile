FROM python:3.12-slim

ENV PYTHONBUFFERED True

ENV APP_HOME /app

WORKDIR $APP_HOME

COPY . ./

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libopencv-dev \
    libsm6 \
    libxext6 \
    libxrender1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install -r requirements.txt
    
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app