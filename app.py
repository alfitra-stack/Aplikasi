import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import model as m

st.set_page_config(page_title="Analisis Harga Pangan Antarprovinsi", layout="wide")


@st.cache_data
def get_data():
    return m.load_data()


@st.cache_data
def get_sil(_prov, komoditas):
    return m.silhouette_table(_prov, komoditas)


prov, panel, komoditas = get_data()

st.title("Analisis Pengelompokan dan Prediksi Harga Bahan Pokok Antarprovinsi")
st.caption("K-Means Clustering + Linear Regression | Data: Harga Pangan Mingguan Jul-Des 2024 "
           "dan UMP 2024 (Satu Data Indonesia) | Kelompok GGS, Teknik Informatika UMRAH")

# ---------------- Sidebar ----------------
st.sidebar.header("Pengaturan")
k = st.sidebar.slider("Jumlah cluster (k)", 2, 6, 3)
st.sidebar.markdown("k = 3 dipilih pada analisis utama (lihat tab Evaluasi untuk Elbow dan Silhouette).")

pc, sil, ev = m.run_kmeans(prov, komoditas, k)
names = m.cluster_names(pc, k)
pc["nama_cluster"] = pc.cluster.map(names)
res, pred, wfull = m.run_regression(panel, pc)

tab1, tab2, tab3, tab4 = st.tabs(["Data", "Clustering", "Prediksi", "Evaluasi"])

# ---------------- Tab 1: Data ----------------
with tab1:
    st.subheader("Indeks Harga Pangan per Provinsi")
    st.write("Indeks 100 = rata-rata harga nasional. Di atas 100 berarti lebih mahal dari rata-rata.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah provinsi", len(prov))
    top = prov.loc[prov.indeks_harga.idxmax()]
    low = prov.loc[prov.indeks_harga.idxmin()]
    c2.metric("Indeks tertinggi", f"{top.indeks_harga:.1f}", top.provinsi)
    c3.metric("Indeks terendah", f"{low.indeks_harga:.1f}", low.provinsi)

    s = prov.sort_values("indeks_harga")
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(s.provinsi, s.indeks_harga, color="#1f4e79")
    ax.axvline(100, color="red", ls="--", lw=1)
    ax.set_xlabel("Indeks Harga Pangan")
    ax.tick_params(axis="y", labelsize=7)
    st.pyplot(fig)

    st.subheader("Data per provinsi (rata-rata Jul-Des 2024)")
    st.dataframe(prov[["provinsi", "ump", "indeks_harga"] + komoditas].round(1),
                 use_container_width=True)

# ---------------- Tab 2: Clustering ----------------
with tab2:
    st.subheader(f"Hasil K-Means (k = {k})")
    st.write(f"Silhouette score: **{sil:.3f}**. Data distandardisasi (Z-score) sebelum clustering.")
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in sorted(pc.cluster.unique()):
        d = pc[pc.cluster == c]
        ax.scatter(d.pc1, d.pc2, s=60, label=names[c])
        for _, r in d.iterrows():
            ax.annotate(r.provinsi, (r.pc1, r.pc2), fontsize=6)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}%)")
    ax.legend(fontsize=8)
    st.pyplot(fig)

    st.subheader("Profil dan anggota cluster")
    prof = pc.groupby("nama_cluster").agg(jumlah=("provinsi", "count"),
                                          indeks_rata2=("indeks_harga", "mean"),
                                          ump_rata2=("ump", "mean")).round(1)
    st.dataframe(prof, use_container_width=True)
    for c in sorted(pc.cluster.unique()):
        st.markdown(f"**{names[c]}**: " + ", ".join(pc[pc.cluster == c].provinsi))
    if pc.cluster.value_counts().min() == 1:
        st.info("Ada cluster yang hanya berisi 1 provinsi. Itu terjadi karena Papua Pegunungan "
                "memiliki harga yang sangat ekstrem (beberapa harganya konstan selama 6 bulan, "
                "kemungkinan harga acuan).")

# ---------------- Tab 3: Prediksi ----------------
with tab3:
    st.subheader("Prediksi Indeks Harga Bulan Berikutnya (Januari 2025)")
    st.write("Model B3: Linear Regression dengan fitur indeks harga bulan ini, UMP, dan cluster provinsi.")
    pilih = st.selectbox("Pilih provinsi", sorted(prov.provinsi))
    row = pred[pred.provinsi == pilih].iloc[0]
    cl_name = names[int(row.cluster)]
    st.markdown(f"**Cluster:** {cl_name}")
    a1, a2 = st.columns(2)
    a1.metric("UMP 2024", f"Rp{row.ump:,.0f}".replace(",", "."))
    a2.metric("Indeks Des 2024", f"{row.indeks_harga:.1f}")
    b1, b2 = st.columns(2)
    b1.metric("Prediksi Jan 2025", f"{row.prediksi_bulan_depan:.1f}", f"{row.perubahan:+.2f}")
    b2.metric("Selisih dari rata-rata nasional", f"{row.prediksi_bulan_depan - 100:+.1f}")

    h = wfull[wfull.provinsi == pilih].sort_values("bulan")
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = list(pd.to_datetime(h.bulan).dt.strftime("%b %Y")) + ["Jan 2025 (prediksi)"]
    ax.plot(labels[:-1], h.indeks_harga, marker="o", label="Aktual")
    ax.plot(labels[-2:], [h.indeks_harga.iloc[-1], row.prediksi_bulan_depan],
            "r--o", label="Prediksi")
    ax.axhline(100, color="grey", ls=":")
    ax.set_ylabel("Indeks Harga Pangan")
    ax.legend()
    plt.xticks(rotation=30)
    st.pyplot(fig)

    st.subheader("Harga komoditas provinsi ini (Jul-Des 2024)")
    komod_pilih = st.multiselect("Komoditas", komoditas, default=["Beras", "Cabai Rawit", "Bawang Merah"])
    if komod_pilih:
        hh = h.set_index(pd.to_datetime(h.bulan).dt.strftime("%Y-%m"))[komod_pilih]
        st.line_chart(hh)

    with st.expander("Prediksi semua provinsi"):
        st.dataframe(pred.sort_values("prediksi_bulan_depan", ascending=False)
                     .round(2), use_container_width=True)

# ---------------- Tab 4: Evaluasi ----------------
with tab4:
    st.subheader("Memilih jumlah cluster")
    sil_df = get_sil(prov, komoditas)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(sil_df.k, sil_df.inertia, "o-"); ax[0].set_title("Elbow Method"); ax[0].set_xlabel("k")
    ax[1].plot(sil_df.k, sil_df.silhouette, "o-", color="green"); ax[1].set_title("Silhouette Score"); ax[1].set_xlabel("k")
    st.pyplot(fig)
    st.dataframe(sil_df.round(3), use_container_width=True)
    st.write("Silhouette tertinggi ada di k = 2 (0,70), tetapi itu hanya memisahkan Papua Pegunungan "
             "dari provinsi lain. k = 3 (0,39) dipilih agar cluster lebih informatif. Pemisahan antar "
             "cluster tidak tegas (silhouette < 0,5).")

    st.subheader("Evaluasi Linear Regression (split waktu)")
    st.write("Data latih: pasangan bulan Jul-Sep ke bulan berikutnya. Data uji: Okt-Nov ke bulan berikutnya.")
    st.dataframe(res.round(3), use_container_width=True)
    r3 = res.loc["B3: Indeks + UMP + cluster", "R2"]
    rb = res.loc["Baseline naif (bulan depan = bulan ini)", "R2"]
    teks = (f"Dengan k = {k}, Model B3 memiliki R2 {r3:.3f}. ")
    if rb > r3:
        teks += (f"Model ini belum mengalahkan baseline naif (harga bulan depan = bulan ini, R2 {rb:.3f}). ")
    teks += ("Harga antarbulan sangat stabil dan data hanya 6 bulan, sehingga prediksi ini bersifat "
             "indikatif, bukan peramalan andal. Hasil analisis utama menggunakan k = 3.")
    st.warning(teks)
