#Use a lightweight Python image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app

# Copy only requirements.txt first for dependency installation
COPY requirements.txt /app/

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg libsm6 libxext6 unzip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy the application source code
COPY src/ /app/src/
COPY app.py /app/

# Copy input data (optional)
COPY notebook/data/ /app/notebook/data

# Create a directory for artifacts
RUN mkdir -p /app/artifacts

# Run data ingestion during build to create pickle files
RUN python /app/src/components/data_ingestion.py

# Expose Flask app port
EXPOSE 5000

# Set the default command to start the Flask app
CMD ["python", "app.py"]
