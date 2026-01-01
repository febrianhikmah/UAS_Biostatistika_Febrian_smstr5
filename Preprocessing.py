def preprocessing_app():
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    import altair as alt
    
    st.title("Preprocessing Data")

    st.markdown(
        """
        **Ini adalah tahapan preprocessing data**, proses ini penting karena data mentah umumnya 
        masih mengandung berbagai permasalahan yang dapat mengganggu kualitas analisis dan 
        kinerja model. Permasalahan tersebut meliputi nilai hilang, inkonsistensi skala antar 
        variabel, keberadaan outlier, noise, serta format data yang belum sesuai dengan 
        kebutuhan analisis.
        """
    )

    # ================= BACA DATA =================
    df = pd.read_excel("data kesehatan2.xlsx")

    st.subheader("Data Awal")
    st.write(df.head())

    # ================= PILIHAN PREPROCESSING =================
    choice = st.radio(
        "Apakah ingin preprocessing data?",
        ("Tidak", "Ya")
    )

    if choice == "Tidak":
        st.info("Preprocessing tidak dilakukan.")
        return

    st.subheader("Tahapan Preprocessing Data")
    col1, col2 = st.columns(2)

    # ================= 1. IMBALANCED DATA =================
    with col1:
        st.markdown("### 1. Cek Data Tidak Seimbang (Imbalanced Data)")
        target_col = "Y_Gallstone Status"
        class_dist = df[target_col].value_counts(normalize=True)

        st.write("Distribusi kelas:")
        st.write(class_dist)

        if class_dist.max() > 0.6:
            st.warning("⚠️ Data TIDAK seimbang")
        else:
            st.success("✅ Data seimbang")

    # ================= 2. MISSING VALUES =================
    with col2:
        st.markdown("### 2. Cek Missing Values")
        missing = df.isnull().sum()
        st.write(missing)

        if missing.sum() == 0:
            st.success("✅ Tidak ada data hilang")
        else:
            st.warning("⚠️ Terdapat data hilang")

    # ================= FUNCTION BOXPLOT =================
    def tampilkan_boxplot(df, num_cols, suffix_key):
        st.write("Distribusi variabel numerik.")

        show_all = st.checkbox(
            "Tampilkan semua variabel",
            value=True,
            key=f"show_all_{suffix_key}"
        )

        if show_all:
            selected_cols = num_cols
        else:
            selected_cols = st.multiselect(
                "Pilih variabel yang ingin ditampilkan:",
                options=num_cols,
                default=list(num_cols[:3]),
                key=f"multiselect_{suffix_key}"
            )

        if len(selected_cols) == 0:
            st.warning("⚠️ Pilih minimal satu variabel untuk ditampilkan.")
            return

        box_df = df[selected_cols].melt(
            var_name="Variabel",
            value_name="Nilai"
        )

        boxplot = alt.Chart(box_df).mark_boxplot().encode(
            x=alt.X("Variabel:N", title="Variabel"),
            y=alt.Y("Nilai:Q", title="Nilai"),
            color=alt.Color("Variabel:N", legend=None),
            tooltip=["Variabel", "Nilai"]
        ).properties(height=400)

        st.altair_chart(boxplot, use_container_width=True)

    # ================= 2. OUTLIER & NOISE =================
    st.markdown("### 3. Penanganan Outlier dan Missing Value")

    exclude_cols = ["Y_Gallstone Status", "X2_Gender", "X3_Comorbidity", "X4_Diabetes Mellitus (DM)"]

    num_cols = [
        col for col in df.select_dtypes(include=np.number).columns
        if col not in exclude_cols
    ]

    # ================= DISTRIBUSI AWAL =================
    st.subheader("Distribusi Variabel Numerik (Sebelum Penanganan Outlier)")
    tampilkan_boxplot(df, num_cols, suffix_key="before")
    # st.write(df)
    
    # ================= DETEKSI & PENANGANAN OUTLIER =================
    outlier_info = {}

    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.2 * IQR
        upper_bound = Q3 + 1.2 * IQR

        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_count = outlier_mask.sum()

        outlier_info[col] = outlier_count

        median = df[col].median()
        df.loc[outlier_mask, col] = median

    st.write("Jumlah outlier pada setiap variabel:")
    st.write(outlier_info)

    if sum(outlier_info.values()) == 0:
        st.success("✅ Tidak ditemukan outlier")
    else:
        st.success("✅ Outlier berhasil ditangani menggunakan median")

    # ================= DISTRIBUSI SETELAH =================
    st.subheader("Distribusi Variabel Numerik (Setelah Penanganan Outlier)")
    tampilkan_boxplot(df, num_cols, suffix_key="after")
    # st.write(df)  

    # ================= 3. CEK SKALA DATA =================
    st.markdown("### 4. Cek Skala Data")

    num_cols = df.select_dtypes(include=np.number).columns

    # Statistik ringkas untuk melihat skala
    scale_df = df[num_cols].agg(["min", "max", "mean", "std"]).T
    scale_df.columns = ["Min", "Max", "Mean", "Std Dev"]

    st.write("Ringkasan statistik variabel numerik:")
    st.dataframe(scale_df, use_container_width=True)

    # Deteksi perbedaan skala
    max_range = scale_df["Max"].max()
    min_range = scale_df["Max"].min()
    range_ratio = max_range / min_range if min_range != 0 else np.inf

    if range_ratio > 10:
        st.warning(
            "⚠️ Terdapat perbedaan skala yang cukup besar antar variabel numerik.\n\n"
            "📌 **Rekomendasi**: "
            "Data sebaiknya dilakukan **normalisasi atau standarisasi** "
            "sebelum digunakan dalam pemodelan, "
            "terutama untuk model yang sensitif terhadap skala seperti "
            "regresi logistik dan metode machine learning."
        )
    else:
        st.success(
            "✅ Skala antar variabel relatif seragam. "
            "Normalisasi data tidak bersifat wajib untuk analisis lanjutan."
        )

    # ================= 4. CEK KORELASI ANTAR VARIABEL X =================
    st.markdown("### 5. Cek Korelasi Antar Variabel Prediktor")

    y = df["Y_Gallstone Status"]
    X = df.drop(columns=["Y_Gallstone Status"])
    X_num = X.select_dtypes(include=np.number)
    

    if X_num.shape[1] < 2:
        st.warning("⚠️ Jumlah variabel numerik kurang dari 2, korelasi tidak dapat dihitung.")
    else:
        corr_matrix = X_num.corr()

        st.write("Matriks korelasi Pearson antar variabel prediktor:")
        st.dataframe(
            corr_matrix.style.format("{:.2f}"),
            use_container_width=True
        )

    # Deteksi pasangan korelasi tinggi
    threshold = 0.7
    high_corr = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                high_corr.append({
                    "Variabel 1": corr_matrix.columns[i],
                    "Variabel 2": corr_matrix.columns[j],
                    "Korelasi": corr_val
                })

    if high_corr:
        high_corr_df = pd.DataFrame(high_corr)
        st.warning(f"⚠️ Ditemukan pasangan variabel dengan korelasi tinggi (|r| ≥ {threshold})")
        st.dataframe(
            high_corr_df.style.format({"Korelasi": "{:.2f}"}),
            use_container_width=True
        )

        st.info(
            "📌 **Rekomendasi**: "
            "Pasangan variabel dengan korelasi tinggi berpotensi menyebabkan "
            "multikolinearitas. Pertimbangkan pemilihan salah satu variabel "
            "atau penggabungan sebelum pemodelan."
        )
    else:
        st.success(
            f"✅ Tidak ditemukan korelasi tinggi antar variabel prediktor "
            f"(seluruh |r| < {threshold})."
        )

    st.markdown("### Seleksi Variabel Prediktor (X)")

    cols_to_drop = st.multiselect(
        "Pilih variabel X yang akan dihapus:",
        options=X_num.columns.tolist()
    )

    if cols_to_drop:
        X_selected = X_num.drop(columns=cols_to_drop)
        st.success(f"✅ Variabel X yang dihapus: {', '.join(cols_to_drop)}")
    else:
        X_selected = X_num.copy()
        st.info("ℹ️ Tidak ada variabel X yang dihapus.")

    df_final = pd.concat([y, X_selected], axis=1)
    
    st.subheader("Dataset Final Setelah Preprocessing")
    st.write(f"Jumlah baris: **{df_final.shape[0]}**")
    st.write(f"Jumlah kolom: **{df_final.shape[1]}**")
    st.dataframe(df_final.head(), use_container_width=True)

    # ================= SIMPAN DATA OTOMATIS =================
    output_file = "data_kesehatan2_preprocessed.xlsx"
    df_final.to_excel(output_file, index=False)  
    st.success(f"💾 Data berhasil disimpan otomatis sebagai **{output_file}**")

# ================= PANGGIL FUNGSI =================
# preprocessing_app()
