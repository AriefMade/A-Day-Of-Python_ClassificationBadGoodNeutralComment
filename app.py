import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize

# Custom preprocessing function (sama dengan yang di train_model.py)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    return ' '.join(tokens)

# Load model dan classes
@st.cache_resource
def load_model():
    model = joblib.load('model/model.pkl')
    try:
        classes = joblib.load('model/classes.pkl')
    except:
        classes = None
    return model, classes

model, classes = load_model()

# UI
st.title("Klasifikasi Sifat dari Komentar")
st.write("Masukkan komentar seseorang, dan sistem akan memprediksi sifatnya.")

komentar_input = st.text_area("Komentar")

# Contoh komentar
with st.expander("Lihat Contoh Komentar"):
    st.markdown("""
    **Contoh komentar positif:**
    - Pelayanannya sangat baik dan ramah, saya sangat puas!
    - Saya merasa bahagia dengan hasil kerja tim ini, luar biasa!
    
    **Contoh komentar negatif:**
    - Produk ini benar-benar mengecewakan, kualitasnya buruk sekali.
    - Pelayanan customer service lambat dan tidak profesional.
    
    **Contoh komentar netral:**
    - Acara ini berlangsung dari jam 8 pagi sampai jam 5 sore.
    - Gedung ini memiliki 10 lantai dan lift di setiap sudutnya.
    """)

if st.button("Klasifikasikan"):
    if komentar_input.strip() != "":
        # Preprocessing komentar
        komentar_preprocessed = preprocess_text(komentar_input)
        
        # Prediksi
        hasil = model.predict([komentar_preprocessed])[0]
        
        # Tampilkan hasil dengan styling berbeda untuk setiap sifat
        if hasil == "positif":
            st.success(f"Sifat yang terdeteksi: **{hasil.upper()}** 😊")
        elif hasil == "negatif":
            st.error(f"Sifat yang terdeteksi: **{hasil.upper()}** 😞")
        else:  # netral
            st.info(f"Sifat yang terdeteksi: **{hasil.upper()}** 😐")
        
        # Tampilkan probabilitas (jika model mendukung)
        try:
            proba = model.predict_proba([komentar_preprocessed])[0]
            proba_df = pd.DataFrame({
                'Sifat': model.classes_,
                'Probabilitas': [f"{p:.2%}" for p in proba]
            })
            st.write("Probabilitas untuk setiap kelas:")
            st.dataframe(proba_df)
        except:
            pass
            
    else:
        st.warning("Masukkan komentar terlebih dahulu.")

# Tambahkan informasi tambahan
st.sidebar.header("Tentang Aplikasi")
st.sidebar.write("""
Aplikasi ini menggunakan model machine learning untuk mengklasifikasikan sifat dari komentar:
- **Positif**: Komentar yang mengandung sentimen positif
- **Negatif**: Komentar yang mengandung sentimen negatif
- **Netral**: Komentar yang bersifat faktual atau tidak memiliki sentimen tertentu
""")

# Tambahkan petunjuk penggunaan
st.sidebar.header("Petunjuk Penggunaan")
st.sidebar.write("""
1. Masukkan komentar di kotak teks
2. Klik tombol "Klasifikasikan"
3. Hasil klasifikasi akan ditampilkan beserta probabilitas untuk setiap kelas
""")