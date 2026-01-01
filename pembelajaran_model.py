def Langkah_model_Klasifikasi():
    import streamlit as st

    st.title("Langkah-Langkah Model Klasifikasi Terbaik")

    st.markdown("""
    Penting untuk memahami bagaimana alur matematis Regresi Logistik bekerja dalam melakukan proses klasifikasi sekaligus sebagai dasar inferensi statistika. Oleh karena itu, pembahasan akan disajikan secara bertahap, mencakup formulasi model, proses estimasi parameter, serta pengujian signifikansi dan interpretasi efek prediktor, yang dijabarkan sebagai berikut:
    """)
    
    # ===============================
    # 1. Pendefinisian Variabel Respon
    # ===============================
    st.subheader("1️. Pendefinisian Variabel Respon (Bernoulli)")

    st.markdown(r"""
    Regresi logistik digunakan ketika variabel respon bersifat **dikotomi**, 
    yaitu hanya memiliki dua kemungkinan nilai.

    Secara matematis, variabel respon didefinisikan sebagai:

    $$Y_i \in \{0,1\}$$

    dengan keterangan:
    - $Y_i = 1$ → kejadian terjadi (**sukses**)
    - $Y_i = 0$ → kejadian tidak terjadi (**gagal**)

    Karena bersifat biner, maka variabel respon $Y_i$ mengikuti distribusi **Bernoulli**:

    $$Y_i \sim \text{Bernoulli}(p_i)$$

    di mana:

    $$p_i = P(Y_i = 1 \mid \mathbf{x}_i)$$

    merupakan probabilitas bahwa observasi ke-$i$ mengalami kejadian (kelas positif)
    berdasarkan vektor fitur $\mathbf{x}_i$.
    
    ---
    """)

    
    # ===============================
    # 2. Spesifikasi Model Logistik
    # ===============================
    st.subheader("2️. Spesifikasi Model Logistik (Link Function)")

    st.markdown(r"""
    Model regresi logistik menghubungkan probabilitas kejadian dengan variabel
    prediktor melalui **fungsi logit**.

    Fungsi logit didefinisikan sebagai:

    $$\text{logit}(p_i) = \ln\left(\frac{p_i}{1 - p_i}\right)$$

    Dengan menggunakan fungsi logit, hubungan antara probabilitas kejadian dan
    prediktor dapat dimodelkan secara linier.

    Model linier dari regresi logistik dituliskan sebagai:

    $$
    \ln\left(\frac{p_i}{1 - p_i}\right)
    = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_k x_{ik}
    $$

    di mana:
    - **β₀** adalah *intersep*
    - **β₁, β₂, …, βₖ** adalah *koefisien regresi*
    - **xᵢ₁, xᵢ₂, …, xᵢₖ** merupakan nilai fitur ke-k untuk observasi ke-i
    
    ---
    """)

    st.markdown(r"""
    ### 3️. Fungsi Probabilitas (Inverse Logit)

    Dari model logit yang telah dibentuk, diperoleh probabilitas terjadinya suatu kejadian sebagai berikut:

    $$
    p_i = \frac{e^{\beta_0 + \sum_{j=1}^{k} \beta_j x_{ij}}}
    {1 + e^{\beta_0 + \sum_{j=1}^{k} \beta_j x_{ij}}}
    $$

    atau dapat dituliskan dalam bentuk fungsi sigmoid:

    $$
    p_i = \frac{1}{1 + e^{-\eta_i}}
    $$

    dengan:

    $$
    \eta_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_k x_{ik}
    $$
    
    ---
    """)
    

    st.markdown(r"""
    ### 4️. Penyusunan Fungsi Likelihood

    Karena variabel respon $Y_i$ mengikuti distribusi **Bernoulli**, maka fungsi likelihood dari model regresi logistik dapat dituliskan sebagai:

    $$
    L(\boldsymbol{\beta}) = \prod_{i=1}^{n} p_i^{y_i} (1 - p_i)^{1 - y_i}
    $$

    di mana:
    - $p_i = P(Y_i = 1 \mid \mathbf{x}_i)$
    - $y_i \in \{0,1\}$

    ---

    ### 5️. Fungsi Log-Likelihood

    Untuk mempermudah proses estimasi parameter, fungsi likelihood ditransformasikan ke dalam bentuk **log-likelihood**, sehingga diperoleh:

    $$
    \ell(\boldsymbol{\beta}) =
    \sum_{i=1}^{n}
    \left[
    y_i \ln(p_i) + (1 - y_i)\ln(1 - p_i)
    \right]
    $$

    Dengan mensubstitusikan $p_i$ dari fungsi logistik (sigmoid), maka fungsi log-likelihood menjadi **fungsi nonlinier terhadap parameter** $\boldsymbol{\beta}$.
    """)

    st.markdown(r"""
    ---
    ### 6️. Estimasi Parameter (Maximum Likelihood Estimation)

    Parameter $\boldsymbol{\beta}$ diperoleh dengan cara memaksimumkan fungsi log-likelihood, yaitu dengan menyelesaikan persamaan:

    $$
    \frac{\partial \ell(\boldsymbol{\beta})}{\partial \boldsymbol{\beta}} = 0
    $$

    Namun, karena fungsi log-likelihood regresi logistik **tidak memiliki solusi analitik tertutup (closed-form)**, maka proses estimasi dilakukan menggunakan **metode iteratif**.

    Metode optimasi yang umum digunakan antara lain:
    - **Newton–Raphson**
    - **Fisher Scoring**
    - **Iteratively Reweighted Least Squares (IRLS)**

    Bentuk umum pembaruan parameter pada metode Newton–Raphson adalah:

    $$
    \boldsymbol{\beta}^{(t+1)} =
    \boldsymbol{\beta}^{(t)} +
    \mathbf{I}^{-1}(\boldsymbol{\beta}^{(t)})
    \mathbf{U}(\boldsymbol{\beta}^{(t)})
    $$

    dengan:
    - $\mathbf{U}(\boldsymbol{\beta})$ adalah **vektor skor (gradien log-likelihood)**
    - $\mathbf{I}(\boldsymbol{\beta})$ adalah **matriks informasi Fisher**
    - $t$ menyatakan iterasi ke-$t$
    ---
    """)

    st.markdown(r"""
    ### 7️. Prediksi dan Klasifikasi

    Setelah parameter model regresi logistik diestimasi, langkah selanjutnya adalah melakukan **prediksi probabilitas** untuk setiap observasi.

    #### Prediksi Probabilitas

    Probabilitas bahwa observasi ke-$i$ termasuk ke dalam kelas $1$ diberikan oleh:

    $$
    \hat{p}_i = P(Y_i = 1 \mid \mathbf{x}_i)
    $$

    dengan $\hat{p}_i \in (0,1)$ merupakan hasil transformasi fungsi logit (sigmoid).

    #### Aturan Klasifikasi

    Berdasarkan nilai probabilitas tersebut, dilakukan klasifikasi menggunakan **nilai ambang (threshold)** $c$, sehingga prediksi kelas $\hat{Y}_i$ ditentukan sebagai berikut:

    $$
    \hat{Y}_i =
    \begin{cases}
    1, & \hat{p}_i \ge c \\
    0, & \hat{p}_i < c
    \end{cases}
    $$

    Nilai ambang yang umum digunakan adalah:

    $$
    c = 0.5
    $$

    Namun, nilai $c$ dapat disesuaikan tergantung pada tujuan analisis, seperti menyeimbangkan **presisi dan recall**, atau meminimalkan kesalahan klasifikasi tertentu.
    
    ---
    """)

    # ==========================================
    # Inferensi Statistik pada Regresi Logistik
    # ==========================================
    st.subheader("Inferensi Statistik pada Regresi Logistik")

    st.markdown(r"""
    Inferensi statistika dalam regresi logistik mencakup berbagai pendekatan,
    seperti estimasi parameter, pengujian signifikansi, interval kepercayaan,
    *goodness of fit*, serta interpretasi efek melalui *odds ratio*.

    Namun, dalam pembelajaran model ini, inferensi yang dikaji dibatasi pada:

    1. **Uji Signifikansi Simultan menggunakan Likelihood Ratio Test (G-Test)**
    2. **Uji Signifikansi Parsial menggunakan Wald Test**
    3. **Interpretasi Odds Ratio (OR)**

    ---
    """)

    # ==========================================
    # 1. Uji Signifikansi Simultan
    # ==========================================
    st.subheader("1️ Uji Signifikansi Simultan (Likelihood Ratio Test / G-Test)")

    st.markdown(r"""
    Uji signifikansi simultan bertujuan untuk menguji apakah seluruh variabel
    prediktor secara bersama-sama berpengaruh terhadap variabel respon.

    ### Hipotesis
    $$
    H_0 : \beta_1 = \beta_2 = \cdots = \beta_p = 0
    $$
    $$
    H_1 : \text{minimal terdapat satu } \beta_j \neq 0
    $$

    ### Statistik Uji
    Statistik uji yang digunakan adalah **Likelihood Ratio Test**, yang dirumuskan
    sebagai berikut:

    $$
    G = -2 \left[ \log L_0 - \log L_1 \right] \sim \chi^2(p)
    $$

    dengan:
    - $L_0$ : likelihood model tanpa variabel prediktor (*null model*)
    - $L_1$ : likelihood model dengan variabel prediktor (*full model*)
    - $p$ : jumlah variabel prediktor

    ### Inferensi
    - Jika nilai *p-value* < 0,05, maka **tolak $H_0$**
    - Model regresi logistik **signifikan secara simultan**
    - Model **layak digunakan** untuk analisis lebih lanjut

    ---
    """)
    
    # ==========================================
    # 2. Uji Signifikansi Parsial (Wald Test)
    # ==========================================
    st.subheader("2️ Uji Signifikansi Parsial (Wald Test)")

    st.markdown(r"""
    Uji signifikansi parsial digunakan untuk menguji pengaruh **masing-masing
    variabel prediktor** terhadap variabel respon dalam model regresi logistik,
    dengan mengasumsikan variabel lain bersifat konstan.

    ### Hipotesis
    Untuk setiap variabel prediktor ke-$j$, hipotesis yang diuji adalah:

    $$
    H_0 : \beta_j = 0
    $$
    $$
    H_1 : \beta_j \neq 0
    $$

    ### Statistik Uji
    Statistik uji yang digunakan adalah **Wald Test**, yang dirumuskan sebagai:

    $$
    W_j =
    \left(
    \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)}
    \right)^2
    \sim \chi^2(1)
    $$

    dengan:
    - $\hat{\beta}_j$ : estimasi koefisien regresi variabel ke-$j$
    - $SE(\hat{\beta}_j)$ : standar error dari estimasi koefisien
    - $\chi^2(1)$ : distribusi chi-square dengan 1 derajat bebas

    ### Kriteria Pengujian
    - Tolak $H_0$ jika *p-value* < 0,05
    - Terima $H_0$ jika *p-value* ≥ 0,05

    ### Makna Inferensial
    - **Variabel signifikan** → variabel berpengaruh secara statistik terhadap
    *log-odds* kejadian batu empedu
    - **Variabel tidak signifikan** → tidak terdapat cukup bukti statistik bahwa
    variabel tersebut berpengaruh terhadap *log-odds*

    ---
    """)

    # ==========================================
    # 3. Interpretasi Odds Ratio (OR)
    # ==========================================
    st.subheader("3️ Interpretasi Odds Ratio (OR)")

    st.markdown(r"""
    Inferensi substantif dalam regresi logistik dilakukan melalui **Odds Ratio (OR)**,
    yang diperoleh dari eksponensial koefisien regresi logistik.

    ### Definisi Odds Ratio
    Odds Ratio untuk variabel prediktor ke-$j$ didefinisikan sebagai:

    $$
    OR_j = e^{\beta_j}
    $$

    Odds Ratio menunjukkan perbandingan peluang (*odds*) terjadinya suatu kejadian
    antara dua kondisi yang berbeda, dengan asumsi variabel lain bersifat konstan.

    ### Makna Odds Ratio
    Interpretasi nilai Odds Ratio adalah sebagai berikut:
    - **$OR > 1$** → variabel meningkatkan peluang terjadinya kejadian
    - **$OR < 1$** → variabel menurunkan peluang terjadinya kejadian
    - **$OR = 1$** → variabel tidak berpengaruh terhadap kejadian

    ### Peran Odds Ratio dalam Inferensi
    Odds Ratio merupakan bagian **paling penting** dalam regresi logistik karena:
    - Memberikan interpretasi yang mudah dipahami secara klinis dan kebijakan
    - Menjelaskan besar dan arah pengaruh variabel prediktor
    - Menjadi dasar pengambilan keputusan berbasis risiko

    Dengan demikian, interpretasi Odds Ratio menjadi kunci utama dalam penarikan
    kesimpulan substantif dari model regresi logistik yang dibangun.

    ---
    """)

    

# Langkah_model_Klasifikasi()    