import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import io
from google.cloud import storage
import os
from dotenv import load_dotenv
from google.api_core.exceptions import NotFound  # Perbaikan import

# Memuat variabel dari .env
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS') #kredensial untuk akses ke google storage dimana model di simpan

app = Flask(__name__)

# Konfigurasi
BUCKET_NAME = 'test-deploy-herbmate'
MODEL_FILENAME = 'model.h5'

# Fungsi untuk mengunduh model dari GCS jika tidak ada di lokal
def download_model_from_gcs():
    local_model_path = f'./{os.path.basename(MODEL_FILENAME)}'

    # Cek apakah model sudah ada di lokal
    if os.path.exists(local_model_path):
        print("Model sudah tersedia di lokal.")
        return local_model_path

    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_FILENAME)

        print("Mengunduh model dari Cloud Storage...")
        blob.download_to_filename(local_model_path)
        print("Model berhasil diunduh.")

        return local_model_path
    except NotFound:
        print(f"Error: Model file '{MODEL_FILENAME}' tidak ditemukan di bucket '{BUCKET_NAME}'.")
        raise
    except Exception as e:
        print(f"Terjadi kesalahan saat mengunduh model: {e}")
        raise

# Unduh model dan muat ke memori
try:
    model_path = download_model_from_gcs()
    model = load_model(model_path)
    print("Model berhasil dimuat ke memori.")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    model = None

print(model)

@app.route('/')
def home():
    return "Selamat datang di API Model!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model tidak tersedia untuk prediksi.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih.'}), 400

    try:
        # Membaca dan memproses gambar
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert('RGB')  # Pastikan format RGB
        img = img.resize((224, 224))  # Mengubah ukuran gambar
        img_array = np.array(img) / 255.0  # Normalisasi piksel
        img_array = np.expand_dims(img_array, axis=0)  # Menambah dimensi batch

        # Melakukan prediksi
        predictions = model.predict(img_array)
        predicted_class = int(np.argmax(predictions, axis=1)[0])

        return jsonify({'predicted_class': predicted_class, 'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan saat memproses prediksi: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
