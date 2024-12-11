# Gunakan image dasar yang berisi Python
FROM python:3.12

# Menambahkan label untuk metadata image
LABEL maintainer="renaldi203904@gmail.com"
LABEL version="1.0"
LABEL description="Flask API for Model Prediction"

# Set working directory di dalam container
WORKDIR /app

# Menyalin file aplikasi ke dalam container
COPY . /app

# Install dependensi
RUN pip install --no-cache-dir -r requirements.txt

# Salin file kredensial ke dalam container
COPY account-service-key.json /app/account-service-key.json

# Menyalin model ke dalam container
COPY model.h5 /app/model.h5

# Set variabel lingkungan untuk Google Cloud Credentials
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/account-service-key.json"

# Expose port untuk Cloud Run (port default adalah 8080)
EXPOSE 8080

# Menjalankan aplikasi Flask
CMD ["python", "app.py"]

