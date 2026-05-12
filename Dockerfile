FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies for Python scientific stack + R + rpy2
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    gfortran \
    curl \
    ca-certificates \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libcurl4-openssl-dev \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    libpng-dev \
    r-base \
    r-base-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# R packages used by this repository
RUN Rscript -e "install.packages(c('CausalModels','adjustedCurves','WeightIt','survival'), repos='https://cloud.r-project.org')"

COPY . /app

EXPOSE 8000 7860

COPY docker/start.sh /app/docker/start.sh
RUN chmod +x /app/docker/start.sh

CMD ["/app/docker/start.sh"]
