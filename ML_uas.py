def ML_klasifikasi():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import pickle

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        ConfusionMatrixDisplay
    )

    import matplotlib.pyplot as plt
    import seaborn as sns
    import altair as alt

    st.title("Machine Learning Klasifikasi Penyakit Batu Empedu")
    st.write("Model yang digunakan: **Logistic Regression**")

    
    def Pilih_proicecing():
        st.subheader("Data yang di gunakan pada analisis ini adalah data yang sudah di lakukan preprocessing")
        
        df = pd.read_excel("data_kesehatan2_preprocessed.xlsx")
        # df2 = pd.read_excel("data_kesehatan2_preprocessed.xlsx")
        
        st.subheader("1️ Dataset yang Digunakan")
        st.dataframe(df.head())

        # =========================
        # Target & Fitur
        # =========================
        y = df["Y_Gallstone Status"]
        X = df.drop(columns=["Y_Gallstone Status"])

        num_cols = X.select_dtypes(include=["int64", "float64"]).columns
        X_num = X[num_cols]

        # =========================
        # Visualisasi Box Plot
        # =========================
        st.subheader("2️ Distribusi Variabel Numerik (Boxplot)")
        st.write("Distribusi variabel numerik.")

        # Opsi tampilkan semua atau pilih manual
        show_all = st.checkbox("Tampilkan semua variabel", value=True)

        if show_all:
            selected_cols = num_cols
        else:
            selected_cols = st.multiselect(
                "Pilih variabel yang ingin ditampilkan:",
                options=num_cols,
                default=num_cols[:3]
            )

        # Guard clause biar nggak error
        if len(selected_cols) == 0:
            st.warning("⚠️ Pilih minimal satu variabel untuk ditampilkan.")
        else:
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
        
        # =========================
        # Korelasi
        # =========================
        st.subheader("3️ Analisis Korelasi")

        corr = df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # =========================
        # Train-Test Split
        # =========================
        st.subheader("4️ Pembagian Data")

        test_size = st.slider("Proporsi data uji:", 0.1, 0.5, 0.3)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        # =========================
        # 4. Normalisasi Min–Max
        # =========================
        st.subheader("5. Normalisasi Data dengan Min–Max Scaling")

        scaler_train = MinMaxScaler()
        X_scaled_train = scaler_train.fit_transform(X_train)
        X_scaled_train = pd.DataFrame(X_scaled_train, columns=num_cols)

        scaler_test = MinMaxScaler()
        X_scaled_test = scaler_test.fit_transform(X_test)
        X_scaled_test = pd.DataFrame(X_scaled_test, columns=num_cols)

        X_train = X_scaled_train
        X_test = X_scaled_test
        
        st.subheader("Dataset Training yang Digunakan")
        st.dataframe(X_train.tail())
        st.write(f"Jumlah data training : {X_train.shape}")
        
        st.subheader("Dataset Testing yang Digunakan")
        st.dataframe(X_test.tail())
        st.write(f"Jumlah data testing  : {X_test.shape}")
        
        # =========================
        # Logistic Regression
        # =========================
        st.subheader("6 Training Model Logistic Regression")

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # =========================
        # Evaluasi Model
        # =========================
        st.subheader("7 Evaluasi Model")

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        col1, col2 = st.columns([2, 1])

        with col1:
            fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
            disp.plot(cmap="Blues", ax=ax_cm)
            st.pyplot(fig_cm)

        with col2:
            st.metric("Akurasi", f"{acc*100:.2f}%")
            st.metric("Precision", f"{prec*100:.2f}%")
            st.metric("Recall", f"{rec*100:.2f}%")
            st.metric("F1-Score", f"{f1*100:.2f}%")
            st.metric("ROC-AUC", f"{roc*100:.2f}%")

        # =========================
        # Feature Importance
        # =========================
        st.subheader("8 Feature Importance (Logistic Regression)")

        importance_df = pd.DataFrame({
            "Fitur": X_num.columns,
            "Importance": np.abs(model.coef_[0])
        }).sort_values(
            by="Importance", ascending=False
        ).reset_index(drop=True)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(
                importance_df.style.format({"Importance": "{:.4f}"}),
                use_container_width=True
            )

        with col2:
            bar = alt.Chart(importance_df).mark_bar().encode(
                x="Importance:Q",
                y=alt.Y("Fitur:N", sort="-x"),
                tooltip=["Fitur", "Importance"]
            ).properties(height=400)

            st.altair_chart(bar, use_container_width=True)

        # =========================
        # Interpretasi
        # =========================
        st.subheader("9 Interpretasi Feature Importance")

        top = importance_df.iloc[0]
        low = importance_df.iloc[-1]

        st.markdown(
            f"""
            - **Fitur paling berpengaruh**: **{top['Fitur']}**
            - **Fitur paling rendah kontribusinya**: **{low['Fitur']}**

            Hal ini menunjukkan bahwa perubahan pada fitur dengan importance tinggi
            memiliki pengaruh besar terhadap probabilitas terjadinya penyakit batu empedu
            menurut model regresi logistik.
            """
        )

        # =========================
        # Simpan Model
        # =========================
        model_bundle = {
            "model": model,
            "scaler": scaler_train,
            "features": X_num.columns.tolist()
        }

        with open("logistic_regression_gallstone_yes_proicecing.pkl", "wb") as f:
            pickle.dump(model_bundle, f)

    def tidak_Pilih_proicecing():
        st.subheader("Data yang di gunakan pada analisis ini adalah data yang tidak di lakukan preprocessing")         
        
        df = pd.read_excel("data kesehatan2.xlsx")

        st.subheader("1️⃣ Dataset yang Digunakan")
        st.dataframe(df.head())

        # =========================
        # Target & Fitur
        # =========================
        y = df["Y_Gallstone Status"]
        X = df.drop(columns=["Y_Gallstone Status"])

        num_cols = X.select_dtypes(include=["int64", "float64"]).columns
        X_num = X[num_cols]

        # =========================
        # Visualisasi Box Plot
        # =========================
        st.subheader("2️⃣ Distribusi Variabel Numerik (Boxplot)")
        st.write("Distribusi variabel numerik.")

        # Opsi tampilkan semua atau pilih manual
        show_all = st.checkbox("Tampilkan semua variabel", value=True)

        if show_all:
            selected_cols = num_cols
        else:
            selected_cols = st.multiselect(
                "Pilih variabel yang ingin ditampilkan:",
                options=num_cols,
                default=num_cols[:3]
            )

        # Guard clause biar nggak error
        if len(selected_cols) == 0:
            st.warning("⚠️ Pilih minimal satu variabel untuk ditampilkan.")
        else:
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

        # =========================
        # Korelasi
        # =========================
        st.subheader("3️⃣Analisis Korelasi")

        corr = df.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # =========================
        # Train-Test Split
        # =========================
        st.subheader("4️⃣ Pembagian Data")

        test_size = st.slider("Proporsi data uji:", 0.1, 0.5, 0.3)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        st.dataframe(X_train.tail())
        st.write(f"Jumlah data training : {X_train.shape}")

        st.dataframe(X_test.tail())
        st.write(f"Jumlah data testing : {X_test.shape}")

        # =========================
        # Logistic Regression
        # =========================
        st.subheader("5️⃣ Training Model Logistic Regression")

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # =========================
        # Evaluasi Model
        # =========================
        st.subheader("6️⃣ Evaluasi Model")

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        col1, col2 = st.columns([2, 1])

        with col1:
            fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
            disp.plot(cmap="Blues", ax=ax_cm)
            st.pyplot(fig_cm)

        with col2:
            st.metric("Akurasi", f"{acc*100:.2f}%")
            st.metric("Precision", f"{prec*100:.2f}%")
            st.metric("Recall", f"{rec*100:.2f}%")
            st.metric("F1-Score", f"{f1*100:.2f}%")
            st.metric("ROC-AUC", f"{roc*100:.2f}%")

        # =========================
        # Feature Importance
        # =========================
        st.subheader("7️⃣ Feature Importance (Logistic Regression)")

        importance_df = pd.DataFrame({
            "Fitur": X_num.columns,
            "Importance": np.abs(model.coef_[0])
        }).sort_values(
            by="Importance", ascending=False
        ).reset_index(drop=True)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(
                importance_df.style.format({"Importance": "{:.4f}"}),
                use_container_width=True
            )

        with col2:
            bar = alt.Chart(importance_df).mark_bar().encode(
                x="Importance:Q",
                y=alt.Y("Fitur:N", sort="-x"),
                tooltip=["Fitur", "Importance"]
            ).properties(height=400)

            st.altair_chart(bar, use_container_width=True)

        # =========================
        # Interpretasi
        # =========================
        st.subheader("8️⃣ Interpretasi Feature Importance")

        top = importance_df.iloc[0]
        low = importance_df.iloc[-1]

        st.markdown(
            f"""
            - **Fitur paling berpengaruh**: **{top['Fitur']}**
            - **Fitur paling rendah kontribusinya**: **{low['Fitur']}**

            Hal ini menunjukkan bahwa perubahan pada fitur dengan importance tinggi
            memiliki pengaruh besar terhadap probabilitas terjadinya penyakit batu empedu
            menurut model regresi logistik.
            """
        )

        # =========================
        # Simpan Model
        # =========================
        model_bundle = {
            "model": model,
            "features": X_num.columns.tolist()
        }

        with open("logistic_regression_gallstone.pkl", "wb") as f:
            pickle.dump(model_bundle, f)

    # ================= PILIHAN PREPROCESSING =================
    choice = st.radio(
        "Apakah anda telah mempreprocessing data?",
        ("Tidak", "Ya"), key="pilihan_ML"
    )

    if choice == "Ya":
        Pilih_proicecing()
        return
    
    tidak_Pilih_proicecing()
    

# ML_klasifikasi()