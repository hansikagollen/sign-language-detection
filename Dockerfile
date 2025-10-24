# Use slim Python image
FROM python:3.9-slim

WORKDIR /app

# Copy only requirements first (faster caching)
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy only necessary scripts
COPY *.py /app/

# Expose port (if using web interface)
EXPOSE 5000

# Run your main script
CMD ["python", "predict.py"]
