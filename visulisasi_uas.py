import streamlit as st
import pandas as pd
import altair as alt

def eda_kesehatan():
    st.subheader("📊 Exploratory Data Analysis (EDA)")

    def tidak_prepro():
        # ================= LOAD DATA =================
        df = pd.read_excel("data kesehatan2.xlsx")
        st.markdown("### 📄 5 Baris Dataset")
        st.write("Berikut merupakan 5 baris pertama dari dataset yang digunakan dalam penelitian ini.")
        st.dataframe(df.head())
        
        # Rename kolom biar rapi
        df = df.rename(columns={
            'Y_Gallstone Status': 'gallstone',
            'X1_Age': 'age',
            'X2_Gender': 'gender',
            'X3_Comorbidity': 'comorbidity',
            'X4_Diabetes Mellitus (DM)': 'dm',
            'X5_Body Mass Index (BMI)': 'bmi',
            'X6_Visceral Fat Area (VFA)': 'vfa',
            'X7_Body Protein Content (Protein) (%)': 'protein',
            'X8_Glucose': 'glucose',
            'X9_Total Cholesterol (TC)': 'tc',
            'X10_Low Density Lipoprotein (LDL)': 'ldl',
            'X11_Triglyceride': 'tg',
            'X12_Alkaline Phosphatase (ALP)': 'alp',
            'X13_Vitamin D': 'vitd'
        })

        num_cols = [
            'age','bmi','vfa','glucose','tc','ldl','tg','alp','vitd'
        ]

        # ================= 1. UNIVARIAT =================
        st.markdown("### 🔹 Distribusi Data (Univariat)")

        x_uni = st.selectbox("Pilih Variabel", num_cols, key="uni")

        hist = alt.Chart(df).mark_bar(opacity=0.75).encode(
            x=alt.X(f'{x_uni}:Q', bin=alt.Bin(maxbins=30), title=x_uni.upper()),
            y=alt.Y('count():Q', title='Jumlah Pasien'),
            tooltip=['count():Q']
        ).properties(height=350)

        st.altair_chart(hist, use_container_width=True)
        st.markdown("#### 📌 Insight")
        st.write(
            f"""
            Visualisasi histogram menunjukkan distribusi variabel **{x_uni.upper()}** pada seluruh pasien.
            Grafik ini membantu mengidentifikasi pola sebaran data, seperti kecenderungan nilai dominan,
            potensi skewness, serta adanya outlier yang dapat memengaruhi analisis lanjutan.
            """
        )

        # ================= 2. BIVARIAT =================
        st.markdown("### 🔹 Perbandingan terhadap Status Gallstone")

        x_bi = st.selectbox("Pilih Variabel", num_cols, key="bi")

        box = alt.Chart(df).mark_boxplot().encode(
            x=alt.X('gallstone:N', title='Status Gallstone'),
            y=alt.Y(f'{x_bi}:Q', title=x_bi.upper()),
            color=alt.Color('gallstone:N', legend=None)
        ).properties(height=350)

        st.altair_chart(box, use_container_width=True)
        st.markdown("#### 📌 Insight")
        st.write(
            f"""
            Boxplot ini membandingkan distribusi **{x_bi.upper()}** antara pasien dengan dan tanpa gallstone.
            Perbedaan median, rentang interkuartil, serta outlier dapat mengindikasikan adanya variasi
            karakteristik **{x_bi.upper()}** berdasarkan status gallstone.
            """
        )

        # ================= 3. KATEGORIK =================
        st.markdown("### 🔹 Variabel Kategorik")

        cat_map = {
            'Gender': 'gender',
            'Comorbidity': 'comorbidity',
            'Diabetes Mellitus': 'dm'
        }

        cat_choice = st.selectbox("Pilih Variabel Kategorik", list(cat_map.keys()))

        bar = alt.Chart(df).mark_bar().encode(
            x=alt.X(f'{cat_map[cat_choice]}:N', title=cat_choice),
            y=alt.Y('count():Q', title='Jumlah Pasien'),
            color=alt.Color('gallstone:N', title='Gallstone'),
            tooltip=['count():Q']
        ).properties(height=350)

        st.altair_chart(bar, use_container_width=True)
        st.markdown("#### 📌 Insight")
        st.write(
            f"""
            Diagram batang menunjukkan distribusi **{cat_choice}** berdasarkan status gallstone.
            Visualisasi ini memudahkan identifikasi perbedaan proporsi pasien dengan dan tanpa gallstone
            pada masing-masing kategori.
            """
        )

        # ================= 4. MULTIVARIAT =================
        st.markdown("### 🔹 Hubungan Antar Variabel (Multivariat)")

        scatter_map = {
            'BMI vs VFA': ('bmi','vfa'),
            'Glucose vs Triglyceride': ('glucose','tg'),
            'LDL vs Total Cholesterol': ('ldl','tc'),
            'Age vs BMI': ('age','bmi')
        }

        sc_choice = st.selectbox("Pilih Hubungan", list(scatter_map.keys()))
        x_sc, y_sc = scatter_map[sc_choice]

        scatter = alt.Chart(df).mark_circle(size=70).encode(
            x=alt.X(f'{x_sc}:Q', title=x_sc.upper()),
            y=alt.Y(f'{y_sc}:Q', title=y_sc.upper()),
            color=alt.Color('gallstone:N', title='Gallstone'),
            tooltip=['age','bmi','glucose','tc','vitd']
        ).interactive().properties(height=400)

        st.altair_chart(scatter, use_container_width=True)
        st.markdown("#### 📌 Insight")
        st.write(
            f"""
            Scatter plot memperlihatkan hubungan antara **{x_sc.upper()}** dan **{y_sc.upper()}**,
            dengan pewarnaan berdasarkan status gallstone.
            Pola sebaran titik dapat memberikan gambaran awal mengenai kecenderungan hubungan
            antar variabel serta potensi perbedaan pola antara kelompok gallstone.
            """
        )

    def ya_prepro():
        import pandas as pd
        import streamlit as st
        import altair as alt

        # ================= LOAD DATA =================
        df = pd.read_excel("data_kesehatan2_preprocessed.xlsx")

        st.markdown("### 📄 5 Baris Dataset")
        st.write("Berikut merupakan 5 baris pertama dari dataset yang digunakan.")
        st.dataframe(df.head())

        # ================= RENAME KOLOM (AMAN) =================
        rename_map = {
            'Y_Gallstone Status': 'gallstone',
            'X1_Age': 'age',
            'X2_Gender': 'gender',
            'X3_Comorbidity': 'comorbidity',
            'X4_Diabetes Mellitus (DM)': 'dm',
            'X5_Body Mass Index (BMI)': 'bmi',
            'X6_Visceral Fat Area (VFA)': 'vfa',
            'X7_Body Protein Content (Protein) (%)': 'protein',
            'X8_Glucose': 'glucose',
            'X9_Total Cholesterol (TC)': 'tc',
            'X10_Low Density Lipoprotein (LDL)': 'ldl',
            'X11_Triglyceride': 'tg',
            'X12_Alkaline Phosphatase (ALP)': 'alp',
            'X13_Vitamin D': 'vitd'
        }

        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # ================= DETEKSI TIPE DATA =================
        target = 'gallstone' if 'gallstone' in df.columns else None

        num_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if target:
            if target in num_cols:
                num_cols.remove(target)
            if target not in cat_cols:
                cat_cols.append(target)

        # ================= VALIDASI =================
        if not num_cols:
            st.warning("⚠️ Tidak ada variabel numerik yang tersedia.")
            return

        if not target:
            st.warning("⚠️ Variabel target (gallstone) tidak ditemukan.")
            return

        # ================= 1. UNIVARIAT =================
        st.markdown("### 🔹 Distribusi Data (Univariat)")

        x_uni = st.selectbox("Pilih Variabel", num_cols)

        hist = alt.Chart(df).mark_bar(opacity=0.75).encode(
            x=alt.X(f'{x_uni}:Q', bin=alt.Bin(maxbins=30), title=x_uni.upper()),
            y=alt.Y('count():Q', title='Jumlah Pasien')
        ).properties(height=350)

        st.altair_chart(hist, use_container_width=True)

        st.markdown("#### 📌 Insight")
        st.write(
            f"""
            Histogram menunjukkan distribusi **{x_uni.upper()}** pada seluruh pasien.
            Visualisasi ini membantu memahami pola sebaran data serta potensi outlier
            yang dapat memengaruhi analisis selanjutnya.
            """
        )

        # ================= 2. BIVARIAT =================
        st.markdown("### 🔹 Perbandingan terhadap Status Gallstone")

        x_bi = st.selectbox("Pilih Variabel", num_cols, key="bi")

        box = alt.Chart(df).mark_boxplot().encode(
            x=alt.X(f'{target}:N', title='Status Gallstone'),
            y=alt.Y(f'{x_bi}:Q', title=x_bi.upper()),
            color=alt.Color(f'{target}:N', legend=None)
        ).properties(height=350)

        st.altair_chart(box, use_container_width=True)

        st.markdown("#### 📌 Insight")
        st.write(
            f"""
            Boxplot ini membandingkan distribusi **{x_bi.upper()}** berdasarkan status gallstone.
            Perbedaan median dan sebaran nilai dapat mengindikasikan variasi karakteristik
            antar kelompok pasien.
            """
        )

        # ================= 3. KATEGORIK =================
        st.markdown("### 🔹 Variabel Kategorik")

        cat_available = [c for c in cat_cols if c != target]

        if cat_available:
            cat_choice = st.selectbox("Pilih Variabel Kategorik", cat_available)

            bar = alt.Chart(df).mark_bar().encode(
                x=alt.X(f'{cat_choice}:N', title=cat_choice.upper()),
                y=alt.Y('count():Q', title='Jumlah Pasien'),
                color=alt.Color(f'{target}:N', title='Gallstone')
            ).properties(height=350)

            st.altair_chart(bar, use_container_width=True)

            st.markdown("#### 📌 Insight")
            st.write(
                f"""
                Diagram batang menunjukkan distribusi **{cat_choice.upper()}**
                berdasarkan status gallstone, sehingga memudahkan analisis perbedaan
                proporsi antar kategori.
                """
            )
        else:
            st.info("Tidak ada variabel kategorik yang tersedia.")
            
        exclude_multivariat = ['gender', 'comorbidity', 'dm']
        num_cols_multi = [
            c for c in num_cols
            if c not in exclude_multivariat
        ]

        # ================= 4. MULTIVARIAT =================
        st.markdown("### 🔹 Hubungan Antar Variabel (Multivariat)")

        if len(num_cols_multi) >= 2:
            x_sc = st.selectbox("Variabel X", num_cols_multi, key="x_sc")
            y_sc = st.selectbox(
                "Variabel Y",
                [c for c in num_cols_multi if c != x_sc],
                key="y_sc"
            )

            scatter = alt.Chart(df).mark_circle(size=70).encode(
                x=alt.X(f'{x_sc}:Q', title=x_sc.upper()),
                y=alt.Y(f'{y_sc}:Q', title=y_sc.upper()),
                color=alt.Color(f'{target}:N', title='Gallstone')
            ).interactive().properties(height=400)

            st.altair_chart(scatter, use_container_width=True)

            st.markdown("#### 📌 Insight")
            st.write(
                f"""
                Scatter plot memperlihatkan hubungan antara **{x_sc.upper()}** dan **{y_sc.upper()}**
                dengan pewarnaan berdasarkan status gallstone.
                Pola sebaran dapat memberikan gambaran awal mengenai kecenderungan hubungan
                antar variabel.
                """
            )
        else:
            st.info(
                "Jumlah variabel numerik kontinu (selain gender, comorbidity, dan DM) "
                "tidak mencukupi untuk analisis multivariat."
            )

    # ================= PILIHAN PREPROCESSING =================
    choice = st.radio(
        "Apakah anda telah mempreprocessing data?",
        ("Tidak", "Ya"), key="pilihan_Visual_cuy"
    )

    if choice == "Ya":
        ya_prepro()
        return
    
    tidak_prepro()

# eda_kesehatan()
