import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import io
from google.cloud import storage
import os
from dotenv import load_dotenv
from google.api_core.exceptions import NotFound

# Memuat variabel dari .env
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

app = Flask(__name__)

# Konfigurasi
BUCKET_NAME = 'herbmate-models'
MODEL_FILENAME = 'Models/model.h5'  # Path di GCS
LOCAL_MODEL_PATH = './model.h5'  # Path model lokal

# Fungsi untuk mengunduh model dari GCS jika tidak ada di lokal
def download_model_from_gcs():
    # Cek apakah model sudah ada di lokal
    if os.path.exists(LOCAL_MODEL_PATH):
        print("Model sudah tersedia di lokal.")
        return LOCAL_MODEL_PATH

    try:
        # Mengunduh model dari Cloud Storage
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_FILENAME)

        print("Mengunduh model dari Cloud Storage...")
        blob.download_to_filename(LOCAL_MODEL_PATH)
        print("Model berhasil diunduh.")

        return LOCAL_MODEL_PATH
    except NotFound:
        print(f"Error: Model file '{MODEL_FILENAME}' tidak ditemukan di bucket '{BUCKET_NAME}'.")
        raise
    except Exception as e:
        print(f"Terjadi kesalahan saat mengunduh model: {e}")
        raise

# Unduh model atau gunakan model lokal
try:
    model_path = download_model_from_gcs()  # Cek lokal dulu, baru download jika tidak ada
    model = load_model(model_path)
    print("Model berhasil dimuat ke memori.")
except Exception as e:
    print(f"Gagal memuat model: {e}")
    model = None


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
