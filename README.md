# Credit Score Classification — Web Deployment

Deployment berbasis web (Streamlit) untuk model klasifikasi credit score
terbaik yang dihasilkan oleh pipeline training (`main.py`).

## 1. Struktur File

```
UAS/
├── config.py
├── preprocessing.py
├── trainer.py
├── evaluator.py
├── main.py
├── inference.py        <-- BARU: kelas inference untuk deployment
├── app.py               <-- BARU: web app (Streamlit)
├── test_cases.py         <-- BARU: skrip uji 3 test case (Poor/Standard/Good)
├── requirements.txt      <-- BARU
└── models/
    ├── best_model.pkl
    ├── preprocessor.pkl
    └── label_encoder.pkl
```

`inference.py`, `app.py`, dan `test_cases.py` taruh sejajar dengan
`config.py` (di folder root project, `UAS/`), karena ketiganya
melakukan `from config import ...`.

## 2. Prasyarat

Pastikan training sudah pernah dijalankan minimal sekali sehingga
`models/best_model.pkl`, `models/preprocessor.pkl`, dan
`models/label_encoder.pkl` sudah terbentuk:

```bash
python main.py
```

## 3. Install Dependency Tambahan

```bash
pip install -r requirements.txt
```

## 4. Menjalankan Web App

```bash
streamlit run app.py
```

Buka browser ke `http://localhost:8501`. Isi form sesuai data
nasabah, lalu klik **Predict Credit Score**. Hasil prediksi (kelas +
grafik probabilitas tiap kelas) akan tampil di bawah form.

## 5. Cara Kerja `inference.py`

`CreditScoreInference`:
1. Memuat `best_model.pkl`, `preprocessor.pkl`, `label_encoder.pkl`.
2. Menerima satu record mentah (dict) dengan field yang sama seperti
   kolom asli di CSV (tanpa `ID`, `Customer_ID`, `Name`, `SSN`,
   `Month`, dan `Credit_Score`).
3. Mereplikasi langkah cleaning & feature engineering yang identik
   dengan `Preprocessing.clean_data()` /
   `Preprocessing.feature_engineering()`:
   - Parsing `Credit_History_Age` ("X Years and Y Months" → total
     bulan)
   - Mengubah `Type_of_Loan` menjadi `Num_Loan_Types`
   - Menghitung rasio-rasio turunan (`Debt_to_Income_Ratio`,
     `EMI_to_Income_Ratio`, dst.)
4. Mentransformasi fitur dengan `preprocessor` (StandardScaler +
   OneHotEncoder) yang sama persis dengan yang dipakai saat training.
5. Memanggil `model.predict()` / `model.predict_proba()`, lalu
   mengembalikan label asli (lewat `label_encoder.inverse_transform`)
   beserta probabilitas tiap kelas.

## 6. Pengujian dengan Test Case per Kelas

Sesuai requirement tugas (test case yang merepresentasikan setiap
kelas), jalankan:

```bash
python test_cases.py
```

Skrip ini berisi 3 test case buatan tangan:
- **case_poor** — profil dengan utang tinggi, banyak keterlambatan
  bayar, credit mix buruk → diekspektasikan diprediksi **Poor**.
- **case_standard** — profil dengan nilai-nilai menengah →
  diekspektasikan diprediksi **Standard**.
- **case_good** — profil dengan pendapatan tinggi, utang rendah,
  riwayat kredit panjang, credit mix baik → diekspektasikan
  diprediksi **Good**.

Output di terminal menampilkan prediksi dan probabilitas tiap kelas
untuk masing-masing test case — ini bisa langsung di-screenshot
sebagai bukti pengujian.

Untuk bukti pengujian via web (sesuai instruksi tugas: "melakukan
screenshot untuk hasil dari test case pada model yang telah
dideploy"), masukkan nilai dari `case_poor`, `case_standard`, dan
`case_good` (di `test_cases.py`) satu per satu ke dalam form
`app.py`, lalu screenshot hasil prediksi + grafik probabilitas untuk
masing-masing kelas.

## 7. Catatan

- Jika nama kolom mentah pada dataset kamu sedikit berbeda dari yang
  diasumsikan di `inference.py`/`app.py` (mengikuti dataset Kaggle
  "Credit Score Classification" standar), sesuaikan key pada dict
  `raw_input` di `app.py` / `test_cases.py` dengan nama kolom asli
  di CSV kamu.
- `preprocessor.transform()` memilih kolom berdasarkan **nama**
  (bukan urutan), jadi urutan field pada dict input tidak masalah
  selama nama dan jumlah kolomnya lengkap.