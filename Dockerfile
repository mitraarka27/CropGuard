# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY app.py .
COPY disease_info.json .
COPY src/ ./src/

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose port
EXPOSE 7860

# Command to run app
CMD ["python", "app.py"]