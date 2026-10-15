"""Logika model (K-Means + Linear Regression) untuk aplikasi harga pangan."""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (silhouette_score, mean_absolute_error,
                             mean_squared_error, r2_score)

BASE = os.path.dirname(os.path.abspath(__file__))
NON_FEATURE = {"provinsi", "bulan", "indeks_harga", "ump", "kode_provinsi",
               "cluster", "indeks_next"}


def load_data():
    prov = pd.read_csv(os.path.join(BASE, "data", "data_provinsi_bersih.csv"))
    panel = pd.read_csv(os.path.join(BASE, "data", "panel_provinsi_bulan_bersih.csv"))
    komoditas = [c for c in prov.columns if c not in NON_FEATURE]
    return prov, panel, komoditas


def silhouette_table(prov, komoditas, ks=range(2, 9)):
    Z = StandardScaler().fit_transform(prov[komoditas])
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Z)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(Z, km.labels_)})
    return pd.DataFrame(rows)


def cluster_names(prov, k):
    """Beri nama cluster berdasarkan rata-rata indeks harga (urut naik)."""
    means = prov.groupby("cluster").indeks_harga.mean().sort_index()
    if k == 3:
        base = ["Harga rendah-moderat", "Harga tinggi", "Harga ekstrem"]
        return {c: f"Cluster {c}: {base[c]}" for c in means.index}
    return {c: f"Cluster {c} (indeks rata-rata {means[c]:.0f})" for c in means.index}


def run_kmeans(prov, komoditas, k=3):
    prov = prov.copy()
    Z = StandardScaler().fit_transform(prov[komoditas])
    km = KMeans(n_clusters=k, n_init=50, random_state=42).fit(Z)
    raw = km.labels_
    order = (pd.Series(prov.indeks_harga.values).groupby(raw).mean()
             .sort_values().index)
    remap = {old: new for new, old in enumerate(order)}
    prov["cluster"] = [remap[x] for x in raw]
    sil = silhouette_score(Z, raw)
    pcs = PCA(2).fit(Z)
    xy = pcs.transform(Z)
    prov["pc1"], prov["pc2"] = xy[:, 0], xy[:, 1]
    return prov, sil, pcs.explained_variance_ratio_


def _metrics(y, p):
    return {"MAE": mean_absolute_error(y, p),
            "RMSE": mean_squared_error(y, p) ** 0.5,
            "R2": r2_score(y, p)}


def _design(df, use_ump=True, use_cluster=True, cols=None):
    X = df[["indeks_harga"]].copy()
    if use_ump:
        X["ump"] = df["ump"]
    if use_cluster:
        for c in sorted(df["cluster"].unique()):
            X[f"cluster_{c}"] = (df["cluster"] == c).astype(float)
    if cols is not None:
        X = X.reindex(columns=cols, fill_value=0.0)
    return X


def run_regression(panel, prov_clustered):
    """Evaluasi model (split waktu) dan latih model akhir untuk prediksi bulan depan."""
    w = panel.drop(columns=[c for c in ["cluster"] if c in panel.columns])
    w = w.merge(prov_clustered[["provinsi", "cluster"]], on="provinsi")
    w = w.sort_values(["provinsi", "bulan"]).copy()
    w["indeks_next"] = w.groupby("provinsi").indeks_harga.shift(-1)
    d = w.dropna(subset=["indeks_next"])
    months = sorted(d.bulan.unique())
    train, test = d[d.bulan <= months[2]], d[d.bulan > months[2]]

    results = {"Baseline naif (bulan depan = bulan ini)":
               _metrics(test.indeks_next, test.indeks_harga)}
    for name, (u, c) in {"B2: Indeks bulan ini": (False, False),
                         "B3: Indeks + UMP + cluster": (True, True)}.items():
        Xtr = _design(train, u, c)
        m = LinearRegression().fit(Xtr, train.indeks_next)
        p = m.predict(_design(test, u, c, Xtr.columns))
        results[name] = _metrics(test.indeks_next, p)
    # model B1: UMP saja
    m1 = LinearRegression().fit(train[["ump"]], train.indeks_next)
    results["B1: UMP saja"] = _metrics(test.indeks_next, m1.predict(test[["ump"]]))
    res = pd.DataFrame(results).T.loc[
        ["B1: UMP saja", "B2: Indeks bulan ini", "B3: Indeks + UMP + cluster",
         "Baseline naif (bulan depan = bulan ini)"]]

    # model akhir: semua pasangan bulan -> prediksi bulan setelah data terakhir
    Xall = _design(d, True, True)
    final = LinearRegression().fit(Xall, d.indeks_next)
    last = w[w.bulan == w.bulan.max()].copy()
    last["prediksi_bulan_depan"] = final.predict(_design(last, True, True, Xall.columns))
    last["perubahan"] = last.prediksi_bulan_depan - last.indeks_harga
    return res, last[["provinsi", "bulan", "cluster", "ump", "indeks_harga",
                      "prediksi_bulan_depan", "perubahan"]], w
