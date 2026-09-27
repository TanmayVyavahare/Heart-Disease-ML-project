FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create required directories
RUN mkdir -p data/raw data/processed artifacts/models artifacts/preprocessing artifacts/plots logs

# Expose the Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
