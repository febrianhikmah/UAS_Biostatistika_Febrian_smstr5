import streamlit as st
import pandas as pd
import numpy as np
def prediksi():
    def app_prediksi_batu_empedu():
        import streamlit as st
        import pandas as pd
        import pickle

        st.title("🩺 Aplikasi Prediksi Risiko Batu Empedu")
        st.caption("Model Regresi Logistik – Data Tidak Terpreprocessing")

        # =========================
        # LOAD MODEL
        # =========================
        with open("logistic_regression_gallstone.pkl", "rb") as f:
            model_bundle = pickle.load(f)

        model = model_bundle["model"]
        features = model_bundle["features"]

        # =========================
        # INPUT USER
        # =========================
        st.subheader("📋 Input Data Pasien")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Usia (Tahun)", 18, 100, 45)
            bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
            vfa = st.number_input("Visceral Fat Area (VFA)", 1.0, 50.0, 20.0)
            protein = st.number_input("Body Protein (%)", 5.0, 25.0, 15.0)
            vitd = st.number_input("Vitamin D", 5.0, 100.0, 30.0)

        with col2:
            gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
            dm = st.selectbox("Diabetes Mellitus", ["Tidak", "Ya"])
            glucose = st.number_input("Glukosa", 50.0, 250.0, 80.0)
            tc = st.number_input("Total Cholesterol", 100.0, 300.0, 180.0)

        with col3:
            comorb = st.number_input("Jumlah Komorbiditas", 0, 5, 1)
            ldl = st.number_input("LDL", 50.0, 300.0, 120.0)
            tg = st.number_input("Trigliserida", 50.0, 300.0, 150.0)
            alp = st.number_input("ALP", 30.0, 200.0, 90.0)


        gender_val = 1 if gender == "Male" else 0
        dm_val = 1 if dm == "Ya" else 0

        # =========================
        # DATAFRAME SESUAI MODEL
        # =========================
        input_df = pd.DataFrame([[
            age,
            gender_val,
            comorb,
            dm_val,
            bmi,
            vfa,
            protein,
            glucose,
            tc,
            ldl,
            tg,
            alp,
            vitd
        ]], columns=features)

        # =========================
        # PREDIKSI
        # =========================
        if st.button("🔍 Prediksi Risiko Batu Empedu"):
            prob = model.predict_proba(input_df)[0, 1]
            pred = model.predict(input_df)[0]

            st.markdown("---")
            st.subheader("📊 Hasil Prediksi")

            if prob < 0.30:
                risk = "🟢 Risiko Rendah"
            elif prob < 0.60:
                risk = "🟡 Risiko Sedang"
            else:
                risk = "🔴 Risiko Tinggi"

            colA, colB = st.columns(2)
            colA.metric("Probabilitas Batu Empedu", f"{prob:.2%}")
            colB.metric("Kategori Risiko", risk)

            if pred == 1:
                st.error("⚠️ Pasien diprediksi BERISIKO mengalami batu empedu.")
            else:
                st.success("✅ Pasien diprediksi risiko rendah batu empedu.")
                
            # =========================
            # TAMPILKAN KATEGORI KLINIS
            # =========================
            st.subheader("🧾 Interpretasi Kategori Klinis")

            def kategori_age(x):
                return "Dewasa Muda" if x < 40 else "Paruh Baya" if x < 60 else "Lansia"

            def kategori_bmi(x):
                if x < 18.5:
                    return "Underweight"
                elif x < 25:
                    return "Normal"
                elif x < 30:
                    return "Overweight"
                else:
                    return "Obesitas"

            def kategori_glucose(x):
                return "Normal" if x < 100 else "Prediabetes" if x < 126 else "Diabetes"

            def kategori_chol(x):
                return "Normal" if x < 200 else "Borderline" if x < 240 else "Tinggi"

            def kategori_ldl(x):
                return "Optimal" if x < 100 else "Borderline" if x < 160 else "Tinggi"

            def kategori_tg(x):
                return "Normal" if x < 150 else "Borderline" if x < 200 else "Tinggi"

            def kategori_vfa(x):
                return "Normal" if x < 100 else "Tinggi"

            def kategori_vitd(x):
                return "Defisiensi" if x < 20 else "Insufisiensi" if x < 30 else "Cukup"

            def kategori_comorb(x):
                return "Tidak Ada" if x == 0 else "Ringan" if x <= 1 else "Sedang" if x <= 2 else "Berat"

            def kategori_dm(x):
                return "Tidak Diabetes" if x == 0 else "Diabetes Mellitus"

            def kategori_gender(x):
                return "Perempuan" if x == 0 else "Laki-laki"

            kategori_df = pd.DataFrame({
                "Variabel": [
                    "Usia", "Jenis Kelamin", "Komorbiditas", "Diabetes Mellitus",
                    "BMI", "Visceral Fat Area",
                    "Protein Tubuh", "Glukosa",
                    "Total Cholesterol", "LDL",
                    "Trigliserida", "ALP", "Vitamin D"
                ],
                "Nilai": [
                    age, gender_val, comorb, dm_val,
                    bmi, vfa,
                    protein, glucose,
                    tc, ldl,
                    tg, alp, vitd
                ],
                "Kategori Klinis": [
                    kategori_age(age),
                    kategori_gender(gender_val),
                    kategori_comorb(comorb),
                    kategori_dm(dm_val),
                    kategori_bmi(bmi),
                    kategori_vfa(vfa),
                    "Normal" if protein >= 14 else "Rendah",
                    kategori_glucose(glucose),
                    kategori_chol(tc),
                    kategori_ldl(ldl),
                    kategori_tg(tg),
                    "Normal" if alp <= 120 else "Tinggi",
                    kategori_vitd(vitd)
                ]
            })

            st.dataframe(kategori_df, use_container_width=True)
            
            def rekomendasi_klinis(
                age, gender, comorb, dm, bmi, vfa, protein,
                glucose, tc, ldl, tg, alp, vitd
            ):
                rekom = []

                # Usia
                if age >= 60:
                    rekom.append("Pasien usia lanjut. Disarankan pemantauan kesehatan rutin dan pengendalian faktor risiko metabolik.")

                # Komorbiditas
                if comorb >= 3:
                    rekom.append("Jumlah komorbiditas tinggi. Diperlukan pengelolaan penyakit penyerta secara terintegrasi.")
                elif comorb >= 1:
                    rekom.append("Terdapat penyakit penyerta. Disarankan kontrol rutin dan kepatuhan terhadap terapi.")

                # Diabetes Mellitus
                if dm == 1:
                    rekom.append("Pasien dengan Diabetes Mellitus. Disarankan kontrol glikemik ketat untuk menurunkan risiko pembentukan batu empedu.")

                # BMI
                if bmi < 18.5:
                    rekom.append("Pasien tergolong underweight. Disarankan evaluasi status gizi dan peningkatan asupan nutrisi seimbang.")
                elif bmi < 25:
                    rekom.append("BMI dalam rentang normal. Pertahankan pola makan seimbang dan aktivitas fisik teratur.")
                elif bmi < 30:
                    rekom.append("Pasien overweight. Dianjurkan pengaturan pola makan dan peningkatan aktivitas fisik.")
                else:
                    rekom.append("Pasien obesitas. Disarankan program penurunan berat badan dan konsultasi gizi.")

                # Visceral Fat Area
                if vfa >= 100:
                    rekom.append("Lemak viseral tinggi. Dianjurkan diet rendah lemak jenuh dan olahraga aerobik rutin.")

                # Protein tubuh
                if protein < 14:
                    rekom.append("Kadar protein tubuh rendah. Disarankan peningkatan asupan protein berkualitas.")

                # Glukosa
                if glucose >= 126:
                    rekom.append("Kadar glukosa darah tinggi. Disarankan pemeriksaan lanjutan dan pengendalian gula darah.")
                elif glucose >= 100:
                    rekom.append("Kadar glukosa borderline. Disarankan modifikasi gaya hidup.")

                # Total Cholesterol
                if tc >= 240:
                    rekom.append("Kolesterol total tinggi. Disarankan diet rendah lemak jenuh dan pemeriksaan lipid lanjutan.")
                elif tc >= 200:
                    rekom.append("Kolesterol borderline. Perlu pemantauan berkala dan pengaturan pola makan.")

                # LDL
                if ldl >= 160:
                    rekom.append("Kadar LDL tinggi. Dianjurkan pengendalian lipid secara ketat.")
                elif ldl >= 100:
                    rekom.append("Kadar LDL borderline. Perlu pengaturan diet dan aktivitas fisik.")

                # Trigliserida
                if tg >= 200:
                    rekom.append("Trigliserida tinggi. Dianjurkan pembatasan asupan gula sederhana dan lemak.")
                elif tg >= 150:
                    rekom.append("Trigliserida borderline. Perlu modifikasi gaya hidup.")

                # ALP
                if alp > 120:
                    rekom.append("Kadar ALP meningkat. Disarankan evaluasi fungsi hepatobilier lebih lanjut.")

                # Vitamin D
                if vitd < 20:
                    rekom.append("Defisiensi Vitamin D. Disarankan suplementasi dan paparan sinar matahari.")
                elif vitd < 30:
                    rekom.append("Kadar Vitamin D belum optimal. Disarankan peningkatan paparan sinar matahari.")

                if not rekom:
                    rekom.append("Kondisi pasien secara umum berada dalam batas normal. Disarankan mempertahankan gaya hidup sehat.")

                return rekom

            st.subheader("💡 Rekomendasi Kesehatan")

            rekomendasi = rekomendasi_klinis(
                age, gender_val, comorb, dm_val, bmi, vfa, protein,
                glucose, tc, ldl, tg, alp, vitd
            )

            for r in rekomendasi:
                st.write("•", r)

            st.info("⚠️ Rekomendasi bersifat edukatif dan tidak menggantikan diagnosis atau konsultasi dokter.")


    def app_prediksi_batu_empedu2():
        import streamlit as st
        import pandas as pd
        import pickle

        st.title("🩺 Aplikasi Prediksi Risiko Batu Empedu")
        st.caption("Model Regresi Logistik – Data sudah Terpreprocessing")

        # =========================
        # LOAD MODEL
        # =========================
        with open("logistic_regression_gallstone_yes_proicecing.pkl", "rb") as f:
            model_bundle = pickle.load(f)

        model = model_bundle["model"]
        scaler = model_bundle["scaler"]
        features = model_bundle["features"]

        # =========================
        # INPUT USER
        # =========================
        st.subheader("📋 Input Data Pasien")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Usia (Tahun)", 18, 100, 45)
            bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
            protein = st.number_input("Body Protein (%)", 5.0, 25.0, 15.0)
            vitd = st.number_input("Vitamin D", 5.0, 100.0, 30.0)

        with col2:
            gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
            dm = st.selectbox("Diabetes Mellitus", ["Tidak", "Ya"])
            glucose = st.number_input("Glukosa", 50.0, 250.0, 80.0)
            tc = st.number_input("Total Cholesterol", 100.0, 300.0, 180.0)

        with col3:
            comorb = st.number_input("Jumlah Komorbiditas", 0, 5, 1)
            tg = st.number_input("Trigliserida", 50.0, 300.0, 150.0)
            alp = st.number_input("ALP", 30.0, 200.0, 90.0)


        gender_val = 1 if gender == "Male" else 0
        dm_val = 1 if dm == "Ya" else 0

        # =========================
        # DATAFRAME SESUAI MODEL
        # =========================
        input_df = pd.DataFrame([[
            age,
            gender_val,
            comorb,
            dm_val,
            bmi,
            protein,
            glucose,
            tc,
            tg,
            alp,
            vitd
        ]], columns=features)

        # =========================
        # PREDIKSI
        # =========================
        if st.button("🔍 Prediksi Risiko Batu Empedu"):

            # 👉 SCALING INPUT (WAJIB)
            input_scaled = scaler.transform(input_df)
            input_scaled = pd.DataFrame(input_scaled, columns=features)

            prob = model.predict_proba(input_scaled)[0, 1]
            pred = model.predict(input_scaled)[0]

            st.markdown("---")
            st.subheader("📊 Hasil Prediksi")


            if prob < 0.30:
                risk = "🟢 Risiko Rendah"
            elif prob < 0.60:
                risk = "🟡 Risiko Sedang"
            else:
                risk = "🔴 Risiko Tinggi"

            colA, colB = st.columns(2)
            colA.metric("Probabilitas Batu Empedu", f"{prob:.2%}")
            colB.metric("Kategori Risiko", risk)

            if pred == 1:
                st.error("⚠️ Pasien diprediksi BERISIKO mengalami batu empedu.")
            else:
                st.success("✅ Pasien diprediksi risiko rendah batu empedu.")
                
            # =========================
            # TAMPILKAN KATEGORI KLINIS
            # =========================
            st.subheader("🧾 Interpretasi Kategori Klinis")

            def kategori_age(x):
                return "Dewasa Muda" if x < 40 else "Paruh Baya" if x < 60 else "Lansia"

            def kategori_bmi(x):
                if x < 18.5:
                    return "Underweight"
                elif x < 25:
                    return "Normal"
                elif x < 30:
                    return "Overweight"
                else:
                    return "Obesitas"

            def kategori_glucose(x):
                return "Normal" if x < 100 else "Prediabetes" if x < 126 else "Diabetes"

            def kategori_chol(x):
                return "Normal" if x < 200 else "Borderline" if x < 240 else "Tinggi"

            def kategori_ldl(x):
                return "Optimal" if x < 100 else "Borderline" if x < 160 else "Tinggi"

            def kategori_tg(x):
                return "Normal" if x < 150 else "Borderline" if x < 200 else "Tinggi"

            def kategori_vfa(x):
                return "Normal" if x < 100 else "Tinggi"

            def kategori_vitd(x):
                return "Defisiensi" if x < 20 else "Insufisiensi" if x < 30 else "Cukup"

            def kategori_comorb(x):
                return "Tidak Ada" if x == 0 else "Ringan" if x <= 1 else "Sedang" if x <= 2 else "Berat"

            def kategori_dm(x):
                return "Tidak Diabetes" if x == 0 else "Diabetes Mellitus"

            def kategori_gender(x):
                return "Perempuan" if x == 0 else "Laki-laki"

            kategori_df = pd.DataFrame({
                "Variabel": [
                    "Usia", "Jenis Kelamin", "Komorbiditas", "Diabetes Mellitus",
                    "BMI",
                    "Protein Tubuh", "Glukosa",
                    "Total Cholesterol",
                    "Trigliserida", "ALP", "Vitamin D"
                ],
                "Nilai": [
                    age, gender_val, comorb, dm_val,
                    bmi, 
                    protein, glucose,
                    tc, 
                    tg, alp, vitd
                ],
                "Kategori Klinis": [
                    kategori_age(age),
                    kategori_gender(gender_val),
                    kategori_comorb(comorb),
                    kategori_dm(dm_val),
                    kategori_bmi(bmi),
                    
                    "Normal" if protein >= 14 else "Rendah",
                    kategori_glucose(glucose),
                    kategori_chol(tc),
                    
                    kategori_tg(tg),
                    "Normal" if alp <= 120 else "Tinggi",
                    kategori_vitd(vitd)
                ]
            })

            st.dataframe(kategori_df, use_container_width=True)
            
            def rekomendasi_klinis(
                age, gender, comorb, dm, bmi, protein,
                glucose, tc, tg, alp, vitd
            ):
                rekom = []

                # Usia
                if age >= 60:
                    rekom.append("Pasien usia lanjut. Disarankan pemantauan kesehatan rutin dan pengendalian faktor risiko metabolik.")

                # Komorbiditas
                if comorb >= 2:
                    rekom.append("Jumlah komorbiditas tinggi. Diperlukan pengelolaan penyakit penyerta secara terintegrasi.")
                elif comorb >= 1:
                    rekom.append("Terdapat penyakit penyerta. Disarankan kontrol rutin dan kepatuhan terhadap terapi.")

                # Diabetes Mellitus
                if dm == 1:
                    rekom.append("Pasien dengan Diabetes Mellitus. Disarankan kontrol glikemik ketat untuk menurunkan risiko pembentukan batu empedu.")

                # BMI
                if bmi < 18.5:
                    rekom.append("Pasien tergolong underweight. Disarankan evaluasi status gizi dan peningkatan asupan nutrisi seimbang.")
                elif bmi < 25:
                    rekom.append("BMI dalam rentang normal. Pertahankan pola makan seimbang dan aktivitas fisik teratur.")
                elif bmi < 30:
                    rekom.append("Pasien overweight. Dianjurkan pengaturan pola makan dan peningkatan aktivitas fisik.")
                else:
                    rekom.append("Pasien obesitas. Disarankan program penurunan berat badan dan konsultasi gizi.")


                # Glukosa
                if glucose >= 126:
                    rekom.append("Kadar glukosa darah tinggi. Disarankan pemeriksaan lanjutan dan pengendalian gula darah.")
                elif glucose >= 100:
                    rekom.append("Kadar glukosa borderline. Disarankan modifikasi gaya hidup.")

                # Total Cholesterol
                if tc >= 240:
                    rekom.append("Kolesterol total tinggi. Disarankan diet rendah lemak jenuh dan pemeriksaan lipid lanjutan.")
                elif tc >= 200:
                    rekom.append("Kolesterol borderline. Perlu pemantauan berkala dan pengaturan pola makan.")

                # ALP
                if alp > 120:
                    rekom.append("Kadar ALP meningkat. Disarankan evaluasi fungsi hepatobilier lebih lanjut.")

                # Vitamin D
                if vitd < 20:
                    rekom.append("Defisiensi Vitamin D. Disarankan suplementasi dan paparan sinar matahari.")
                elif vitd < 30:
                    rekom.append("Kadar Vitamin D belum optimal. Disarankan peningkatan paparan sinar matahari.")

                if not rekom:
                    rekom.append("Kondisi pasien secara umum berada dalam batas normal. Disarankan mempertahankan gaya hidup sehat.")

                return rekom

            st.subheader("💡 Rekomendasi Kesehatan")

            rekomendasi = rekomendasi_klinis(
                age, gender_val, comorb, dm_val, bmi,  protein,
                glucose, tc, tg, alp, vitd
            )

            for r in rekomendasi:
                st.write("•", r)

            st.info("⚠️ Rekomendasi bersifat edukatif dan tidak menggantikan diagnosis atau konsultasi dokter.")


    # ================= PILIHAN PREPROCESSING =================
    choice = st.radio(
        "Apakah anda telah mempreprocessing data?",
        ("Tidak", "Ya"), key="Prediksi_ML"
    )

    if choice == "Ya":
        app_prediksi_batu_empedu2()
        return

    app_prediksi_batu_empedu()

# app_prediksi_batu_empedu2()