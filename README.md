# Flower Federated Learning Simulation

Aplikasi Streamlit untuk demonstrasi konsep Federated Learning dengan satu server dan tiga client.

## Fitur
- Visualisasi topologi 1 server dan 3 client
- Simulasi beberapa ronde pelatihan lokal
- Agregasi berbobot dengan konsep Federated Averaging (FedAvg)
- Grafik akurasi dan loss global
- Tabel aktivitas client dan riwayat model global
- Unduh hasil simulasi dalam CSV

## Menjalankan secara lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud
1. Upload seluruh isi repository ini ke GitHub.
2. Buka https://share.streamlit.io/
3. Login menggunakan GitHub.
4. Pilih **Create app** dan repository proyek ini.
5. Pilih branch `main` dan file utama `app.py`.
6. Klik **Deploy**.

## Catatan penting
Aplikasi ini adalah simulasi visual dan numerik konsep Flower/FedAvg. Aplikasi belum menjalankan proses federated learning terdistribusi menggunakan Flower runtime atau Flower Dashboard. Hasil akurasi dan loss adalah data simulasi untuk keperluan demonstrasi, bukan hasil eksperimen model nyata.
