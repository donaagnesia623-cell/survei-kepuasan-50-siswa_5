import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(page_title="Dashboard 20 Soal", layout="wide")
st.title("📊 Dashboard Analisis Data Siswa (20 Soal)")
st.markdown("Tugas Mata Kuliah Fisika Komputasi")

# ==========================================================
# LOAD DATA
# ==========================================================
@st.cache_data
def load_data():
    df = pd.read_excel("data_simulasi_50_siswa_20_soal.xlsx")
    df = df.iloc[0:50]  # Ambil 50 responden pertama
    return df

try:
    df = load_data()
    st.success("✅ Data 50 responden x 20 soal berhasil dimuat!")
    
    with st.expander("📋 Lihat Data Mentah"):
        st.dataframe(df)
    
    # ==========================================================
    # PERSIAPAN DATA - 20 SOAL
    # ==========================================================
    data_20 = df.iloc[:, 1:21].copy()
    data_20 = data_20.apply(pd.to_numeric, errors='coerce')
    data_20.columns = [f'Soal {i}' for i in range(1, 21)]
    
    # ==========================================================
    # RINGKASAN DATA
    # ==========================================================
    st.header("📌 Ringkasan Data 20 Soal")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Jumlah Responden", len(data_20))
    with col2:
        st.metric("Rata-rata Skor", f"{data_20.mean().mean():.2f}")
    with col3:
        st.metric("Skor Tertinggi", int(data_20.max().max()))
    with col4:
        st.metric("Skor Terendah", int(data_20.min().min()))
    
    st.divider()
    
    # ==========================================================
    # 1️⃣ STATISTIK DESKRIPTIF
    # ==========================================================
    st.header("1️⃣ Statistik Deskriptif 20 Soal")
    
    mean_scores = data_20.mean()
    soal_tertinggi = mean_scores.idxmax()
    soal_terendah = mean_scores.idxmin()
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🏆 **Tertinggi:** {soal_tertinggi} = {mean_scores.max():.2f}")
    with col2:
        st.error(f"📉 **Terendah:** {soal_terendah} = {mean_scores.min():.2f}")
    
    # Grafik 20 soal
    st.subheader("Rata-rata Skor 20 Soal")
    fig, ax = plt.subplots(figsize=(16, 5))
    
    colors = []
    for soal in mean_scores.index:
        if soal == soal_terendah:
            colors.append('red')
        elif soal == soal_tertinggi:
            colors.append('green')
        else:
            colors.append('steelblue')
    
    ax.bar(mean_scores.index, mean_scores.values, color=colors)
    ax.set_ylabel('Rata-rata Skor')
    ax.set_ylim(0, 4)
    ax.set_xticklabels(mean_scores.index, rotation=90)
    st.pyplot(fig)
    
    with st.expander("📊 Tabel Statistik Lengkap"):
        st.dataframe(data_20.describe())
    
    st.divider()
    
    # ==========================================================
    # 2️⃣ 5 SOAL PRIORITAS
    # ==========================================================
    st.header("2️⃣ 5 Soal Prioritas Perbaikan")
    
    soal_prioritas = mean_scores.nsmallest(5)
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(soal_prioritas.index, soal_prioritas.values, color='coral')
    ax2.set_ylabel('Rata-rata Skor')
    ax2.set_ylim(0, 4)
    ax2.set_title('5 Soal dengan Skor Terendah')
    st.pyplot(fig2)
    
    for i, (soal, nilai) in enumerate(soal_prioritas.items(), 1):
        st.write(f"{i}. **{soal}** = {nilai:.2f}")
    
    st.divider()
    
    # ==========================================================
    # 3️⃣ ANALISIS GAP
    # ==========================================================
    st.header("3️⃣ Analisis GAP")
    
    gap = 4 - mean_scores
    fig3, ax3 = plt.subplots(figsize=(16, 5))
    ax3.bar(gap.index, gap.values, color='orange')
    ax3.set_ylabel('Nilai Gap')
    ax3.set_xticklabels(gap.index, rotation=90)
    st.pyplot(fig3)
    
    st.warning(f"📌 **Prioritas perbaikan:** {gap.idxmax()}")
    st.divider()
    
    # ==========================================================
    # 4️⃣ SEGMENTASI
    # ==========================================================
    st.header("4️⃣ Segmentasi Siswa")
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_20.fillna(data_20.mean()))
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data_scaled)
    
    data_cluster = data_20.copy()
    data_cluster['Cluster'] = clusters
    
    cluster_mean = data_cluster.groupby('Cluster').mean()
    st.dataframe(cluster_mean)
    st.divider()
    
    # ==========================================================
    # 5️⃣ REKOMENDASI
    # ==========================================================
    st.header("5️⃣ Rekomendasi")
    
    st.info(f"✨ **Kelebihan:** {soal_tertinggi} ({mean_scores.max():.2f})")
    st.error(f"⚠️ **Perlu Perbaikan:** {soal_terendah} ({mean_scores.min():.2f})")
    st.success("✅ Analisis 20 soal selesai!")

except Exception as e:
    st.error(f"❌ Error: {e}")

print("✅ File dashboard_saya.py berhasil dibuat!")
