import streamlit as st
import pandas as pd
import numpy as np

st.header('Klasifikasi Penyakit Batu Empedu Menggunakan Regresi Logistik Biner')
st.write('**Dosen Pengampu:** Bapak Alwan Fadlurohman, S.Stat., M.Stat ')
st.write('**Nama :** Febrian Hikmah Nur Rohim')
st.write('**Nim :** B2D023016')
st.write('**S1 Sains Data** - Universitas Muhammadiyah Semarang')
st.write('**Semarang, 01 Januari 2026**')
st.markdown(r"""
            Dalam aplikasi ini, menu yang ditampilkan terdiri dari: **(1) About Dataset, (2) Tinjauan Pustaka, (3) Preprocessing, (4) Dashboard, (5) Inferensi Statistika, (6) Machine Learning, (7) Prediction App, dan (8) Contact Me**
            
            ---
            
            """)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                            '1 About Dataset',
                            '2 Tinjauan Pustaka',
                            '3 Preprocessing', 
                            '4 Dashboard', 
                            '5 Inferensi Statistika',
                            '6 Machine Learning',
                            '7 Prediction App',
                            '8 Contact Me'])

with tab1:
    import tentang_data
    tentang_data.about_dataset()

with tab2:
    import pembelajaran_model
    pembelajaran_model.Langkah_model_Klasifikasi()
    
with tab3:
    import Preprocessing
    Preprocessing.preprocessing_app()

with tab4:
    import visulisasi_uas
    visulisasi_uas.eda_kesehatan()
    
with tab5:
    import Inferensi_Statistika
    Inferensi_Statistika.uji_diagnostik()

with tab6:
    import ML_uas
    ML_uas.ML_klasifikasi()

with tab7:
    import prediksi
    prediksi.prediksi()
    
with tab8:
    import gua
    gua.contact_me()
