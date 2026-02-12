FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_THREAD_LIMIT=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    git \
    ca-certificates \
    wget \
    libjpeg62-turbo-dev \
    libpng-dev \
    libtiff-dev \
    zlib1g-dev \
    libwebp-dev \
    libopenjp2-7-dev \
    libarchive-dev \
    libpango1.0-dev \
    libcairo2-dev \
    libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*

ARG TESSERACT_VERSION=5.5.2
RUN git clone --branch "${TESSERACT_VERSION}" --depth 1 https://github.com/tesseract-ocr/tesseract.git /tmp/tesseract \
    && cmake -S /tmp/tesseract -B /tmp/tesseract/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/tesseract/build -j"$(nproc)" \
    && cmake --install /tmp/tesseract/build \
    && ldconfig \
    && rm -rf /tmp/tesseract

RUN mkdir -p /usr/local/share/tessdata \
    && wget -O /usr/local/share/tessdata/heb.traineddata \
    https://github.com/tesseract-ocr/tessdata_best/raw/main/heb.traineddata
RUN ln -sf /usr/local/share/tessdata/heb.traineddata /usr/local/share/heb.traineddata

ENV TESSDATA_PREFIX=/usr/local/share/tessdata/
RUN tesseract --version

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY src ./src
COPY text ./text
COPY text_type.csv ./text_type.csv

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "gunicorn --chdir src --workers 1 --worker-tmp-dir /dev/shm --timeout 240 --graceful-timeout 30 --bind 0.0.0.0:${PORT} web_app:app"]
