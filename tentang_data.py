import streamlit as st

def about_dataset():
    st.write("**Tentang Dataset**")

    # =======================
    # BARIS 1 : GAMBAR
    # =======================
    col1, col2 = st.columns([4, 6])

    with col1:
        st.image(
            "https://keslan.kemkes.go.id/img/bg-img/gambarartikel_1701676464_525951.jpg",
            caption="Penyakit Batu Empedu",
            use_container_width=True
        )

    with col2:
        st.subheader("Dataset Klasifikasi Penyakit Batu Empedu")

    # =======================
    # BARIS 2 : TEKS (FULL WIDTH)
    # =======================
    st.markdown("""
https://archive.ics.uci.edu/dataset/1150/gallstone-1

Dataset klinis ini dikumpulkan dari Poliklinik Rawat Jalan Penyakit Dalam Rumah Sakit Ankara VM Medical Park dan mencakup data dari 319 individu pada periode Juni 2022 hingga Juni 2023, di mana 161 di antaranya didiagnosis menderita penyakit batu empedu. Dataset ini terdiri dari 38 fitur yang mencakup data demografis, bioimpedansi, dan data laboratorium, serta telah memperoleh persetujuan etik dari Komite Etik Rumah Sakit Kota Ankara (E2-23-4632).

Variabel demografis meliputi usia, jenis kelamin, tinggi badan, berat badan, dan indeks massa tubuh (BMI). Data bioimpedansi mencakup air tubuh total, air ekstraseluler dan intraseluler, massa otot dan lemak, protein, luas lemak viseral, serta lemak hati. Fitur laboratorium meliputi glukosa, kolesterol total, HDL, LDL, trigliserida, AST, ALT, ALP, kreatinin, laju filtrasi glomerulus (GFR), CRP, hemoglobin, dan vitamin D.

Dalam penelitian ini, variabel yang digunakan sebagai berikut: variabel respon (Y) dan variabel prediktor (X) yang diperoleh dari data klinis pasien. Variabel respon pada penelitian ini adalah status penyakit batu empedu (Y_Gallstone Status), sedangkan variabel prediktor mencakup faktor demografis, kondisi klinis, serta indikator biokimia yang diduga berhubungan dengan kejadian penyakit batu empedu. Analisis eksploratif data (Exploratory Data Analysis/EDA) dilakukan untuk memahami karakteristik data serta peran masing-masing variabel dalam penelitian.

Variabel X1_Age merepresentasikan usia pasien, sedangkan X2_Gender menunjukkan jenis kelamin. Kedua variabel ini digunakan untuk melihat pengaruh faktor demografis terhadap risiko penyakit batu empedu. Variabel X3_Comorbidity dan X4_Diabetes Mellitus (DM) menggambarkan kondisi kesehatan penyerta yang dimiliki pasien, yang dapat memengaruhi metabolisme dan meningkatkan risiko pembentukan batu empedu.

Variabel X5_Body Mass Index (BMI) dan X6_Visceral Fat Area (VFA) digunakan untuk merepresentasikan status gizi dan distribusi lemak tubuh. Nilai BMI dan lemak viseral yang tinggi diketahui berkaitan dengan gangguan metabolisme lipid yang berkontribusi terhadap pembentukan batu empedu. Variabel X7_Body Protein Content (%) mencerminkan komposisi tubuh dan status nutrisi pasien.

Variabel biokimia meliputi X8_Glucose, X9_Total Cholesterol (TC), X10_Low Density Lipoprotein (LDL), dan X11_Triglyceride, yang digunakan untuk mengevaluasi kondisi metabolisme glukosa dan lipid. Peningkatan kadar variabel-variabel ini berpotensi meningkatkan risiko penyakit batu empedu. Variabel X12_Alkaline Phosphatase (ALP) digunakan sebagai indikator fungsi hepatobilier, sedangkan X13_Vitamin D merepresentasikan status vitamin yang diduga memiliki keterkaitan dengan metabolisme kalsium dan kolesterol.

    """)

# about_dataset()