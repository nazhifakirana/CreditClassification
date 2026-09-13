# Credit Score Classification — Web Deployment

Deployment berbasis web (Streamlit) untuk model klasifikasi credit score
terbaik yang dihasilkan oleh pipeline training (`main.py`).

## 1. Struktur File

```
creditclassification/
├── config.py
├── preprocessing.py
├── trainer.py
├── evaluator.py
├── main.py
├── inference.py        
├── app.py               
├── test_cases.py        
├── requirements.txt      
└── models/
    ├── best_model.pkl
    ├── preprocessor.pkl
    └── label_encoder.pkl
```


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

