import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(page_title="Dashboard 20 Soal", layout="wide")
st.title("📊 Dashboard Analisis Data 20 Soal")
st.markdown("Tugas Mata Kuliah Fisika Komputasi")

# ==========================================================
# LOAD DATA
# ==========================================================
@st.cache_data
def load_data():
    df = pd.read_excel("data_simulasi_50_siswa_20_soal.xlsx")
    # Ambil 50 responden pertama, hapus baris kosong
    df = df.iloc[0:50].dropna(how='all')
    return df

try:
    df = load_data()
    st.success("✅ Data 50 responden x 20 soal berhasil dimuat!")
    
    # ==========================================================
    # SIDEBAR
    # ==========================================================
    st.sidebar.header("🔍 Menu")
    halaman = st.sidebar.radio("Pilih Halaman:", ["Ringkasan", "Analisis Soal", "Korelasi", "Segmentasi"])
    
    # ==========================================================
    # PERSIAPAN DATA
    # ==========================================================
    data_soal = df.iloc[:, 1:21].copy()
    data_soal.columns = [f'Soal {i}' for i in range(1, 21)]
    
    # ==========================================================
    # HALAMAN RINGKASAN
    # ==========================================================
    if halaman == "Ringkasan":
        st.header("📌 Ringkasan Data")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Jumlah Responden", len(data_soal))
        with col2:
            st.metric("Jumlah Soal", 20)
        with col3:
            st.metric("Rata-rata Total", f"{data_soal.mean().mean():.2f}")
        with col4:
            st.metric("Standar Deviasi", f"{data_soal.std().mean():.2f}")
        
        st.divider()
        
        # Tabel statistik
        st.subheader("Statistik Deskriptif")
        st.dataframe(data_soal.describe())
        
        # Grafik rata-rata
        st.subheader("Rata-rata Nilai per Soal")
        fig, ax = plt.subplots(figsize=(15, 5))
        mean_vals = data_soal.mean()
        ax.bar(mean_vals.index, mean_vals.values, color='skyblue')
        ax.set_ylabel('Rata-rata')
        ax.set_ylim(0, 4)
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
    
    # ==========================================================
    # HALAMAN ANALISIS SOAL
    # ==========================================================
    elif halaman == "Analisis Soal":
        st.header("📊 Analisis per Soal")
        
        pilihan = st.selectbox("Pilih Soal:", data_soal.columns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Statistik")
            st.write(f"Mean: {data_soal[pilihan].mean():.2f}")
            st.write(f"Median: {data_soal[pilihan].median():.2f}")
            st.write(f"Std Dev: {data_soal[pilihan].std():.2f}")
            st.write(f"Min: {data_soal[pilihan].min()}")
            st.write(f"Max: {data_soal[pilihan].max()}")
        
        with col2:
            st.subheader("Distribusi")
            fig, ax = plt.subplots()
            data_soal[pilihan].value_counts().sort_index().plot(kind='bar', ax=ax)
            ax.set_xlabel('Nilai')
            ax.set_ylabel('Frekuensi')
            st.pyplot(fig)
    
    # ==========================================================
    # HALAMAN KORELASI
    # ==========================================================
    elif halaman == "Korelasi":
        st.header("🔗 Korelasi Antar Soal")
        
        if st.button("Tampilkan Heatmap"):
            fig, ax = plt.subplots(figsize=(14, 10))
            corr = data_soal.corr()
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, ax=ax)
            ax.set_title('Matriks Korelasi 20 Soal')
            st.pyplot(fig)
    
    # ==========================================================
    # HALAMAN SEGMENTASI
    # ==========================================================
    else:
        st.header("👥 Segmentasi Siswa")
        
        # Clustering
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_soal.fillna(data_soal.mean()))
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster = kmeans.fit_predict(data_scaled)
        
        # Hasil clustering
        df_cluster = data_soal.copy()
        df_cluster['Cluster'] = cluster
        
        st.subheader("Rata-rata per Cluster")
        st.dataframe(df_cluster.groupby('Cluster').mean())
        
        # Visualisasi
        fig, ax = plt.subplots(figsize=(10, 5))
        df_cluster.groupby('Cluster').mean().T.plot(ax=ax)
        ax.set_title('Profil Setiap Cluster')
        ax.set_xlabel('Soal')
        ax.set_ylabel('Rata-rata')
        ax.legend(title='Cluster')
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

except Exception as e:
    st.error(f"❌ Error: {e}")
    st.info("Pastikan file Excel ada di folder yang sama")
