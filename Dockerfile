FROM python:3.9-slim

# Set the working directory
WORKDIR api

# Install necessary build tools and libraries via apt
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libffi-dev \
    libssl-dev \
    cython3 \
    && rm -rf /var/lib/apt/lists/*

# Use conda to install Cython and other Python-related dependencies
# RUN conda install -y cython

# Upgrade pip, setuptools, and wheel
# RUN pip3 install --upgrade pip setuptools wheel

# Copy requirements.txt and install Python dependencies via pip
COPY requirements.txt .
COPY .env .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY api api/
# COPY bot bot/

ENV FLASK_APP=api
ENV FLASK_ENV=development
EXPOSE 5000

# Command to run the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]

