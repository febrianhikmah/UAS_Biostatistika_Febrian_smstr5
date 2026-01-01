import streamlit as st
import pandas as pd
import altair as alt

def eda_kesehatan():
    st.subheader("📊 Exploratory Data Analysis (EDA)")

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

    # ================= 2. BIVARIAT =================
    st.markdown("### 🔹 Perbandingan terhadap Status Gallstone")

    x_bi = st.selectbox("Pilih Variabel", num_cols, key="bi")

    box = alt.Chart(df).mark_boxplot().encode(
        x=alt.X('gallstone:N', title='Status Gallstone'),
        y=alt.Y(f'{x_bi}:Q', title=x_bi.upper()),
        color=alt.Color('gallstone:N', legend=None)
    ).properties(height=350)

    st.altair_chart(box, use_container_width=True)

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

# eda_kesehatan()
