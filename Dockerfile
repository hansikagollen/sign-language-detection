# Use prebuilt TensorFlow image with Python 3.9
FROM tensorflow/tensorflow:2.13.0

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy Python dependency file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app source code
COPY . .

# Expose the port your app will use
EXPOSE 5000

# Start the app
CMD ["python", "predict.py"]
