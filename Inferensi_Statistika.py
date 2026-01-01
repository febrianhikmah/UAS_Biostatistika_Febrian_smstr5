def uji_diagnostik():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    from scipy.stats import chi2
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    st.title("Inferensi Statistika Regresi Logistik")
    st.write("Uji diagnostik model dilakukan secara simultan dan parsial.")
    
    def menggunakan_prepro():
            # =========================
            # Load Data
            # =========================
            df = pd.read_excel("data_kesehatan2_preprocessed.xlsx")

            st.subheader("Dataset yang Digunakan")
            st.dataframe(df.head())

            # =========================
            # Target & Prediktor
            # =========================
            y = df["Y_Gallstone Status"]
            X = df.drop(columns=["Y_Gallstone Status"])

            # Ambil variabel numerik saja
            X = X.select_dtypes(include=["int64", "float64"])
            num_cols = X.select_dtypes(include=["int64", "float64"]).columns
            
            test_size = st.slider(
            "Proporsi data uji:", 0.1, 0.5, 0.3, key="test_size_prepro")
            
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
            scaler = MinMaxScaler()

            X_scaled_train = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index      # 🔥 INI KUNCI
            )

            X_scaled_test = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index       # 🔥 INI JUGA
            )

            X_train = X_scaled_train
            X_test = X_scaled_test


            # Tambah konstanta (intercept)
            X_const = sm.add_constant(X_train)

            # =========================
            # Fit Model Logistik
            # =========================
            logit_model = sm.Logit(y_train, X_const)
            result = logit_model.fit(disp=False)

            # =========================
            # 1️⃣ UJI SIMULTAN (Likelihood Ratio Test)
            # =========================
            st.subheader("1️⃣ Uji Signifikansi Simultan (Likelihood Ratio Test)")

            ll_null = result.llnull
            ll_model = result.llf

            LR_stat = 2 * (ll_model - ll_null)
            df_lr = result.df_model
            p_value_lr = chi2.sf(LR_stat, df_lr)

            st.write(f"Nilai Likelihood Ratio (LR) = **{LR_stat:.4f}**")
            st.write(f"Derajat bebas = **{df_lr}**")
            st.write(f"p-value = **{p_value_lr:.4f}**")

            if p_value_lr < 0.05:
                st.success(
                    "Kesimpulan: Model regresi logistik **signifikan secara simultan**, "
                    "artinya seluruh variabel prediktor secara bersama-sama "
                    "berpengaruh terhadap status penyakit batu empedu."
                )
            else:
                st.warning(
                    "Kesimpulan: Model regresi logistik **tidak signifikan secara simultan**."
                )

            # =========================
            # 2️⃣ UJI PARSIAL (WALD TEST)
            # =========================
            st.subheader("2️⃣ Uji Signifikansi Parsial (Wald Test)")

            summary_df = pd.DataFrame({
                "Koefisien": result.params,
                "Std Error": result.bse,
                "Z-Stat": result.tvalues,
                "p-value": result.pvalues
            })
            
            # =========================
            # FILTER SIGNIFIKAN
            # =========================
            alpha = 0.05
            signif_df = summary_df[summary_df["p-value"] < alpha]

            if signif_df.empty:
                st.warning("⚠️ Tidak terdapat variabel yang signifikan secara parsial (p-value < 0,05).")
            else:
                st.markdown("### 🔎 Variabel yang Signifikan secara Parsial (p-value < 0,05)")
                st.dataframe(
                    signif_df.style.format({
                        "Koefisien": "{:.4f}",
                        "Std Error": "{:.4f}",
                        "Z-Stat": "{:.4f}",
                        "p-value": "{:.4f}"
                    }),
                    use_container_width=True
                )

            st.write(
                """
                **Kriteria pengujian**:
                - H₀: βᵢ = 0 (variabel tidak berpengaruh)
                - H₁: βᵢ ≠ 0 (variabel berpengaruh)
                - Tolak H₀ jika p-value < 0,05
                """
            )

            # =========================
            # 3️⃣ ODDS RATIO (SIGNIFIKAN SAJA)
            # =========================
            st.subheader("3️⃣ Odds Ratio (α = 0.05)")

            alpha = 0.05

            odds_ratio = np.exp(result.params)
            conf = result.conf_int()
            conf.columns = ["Lower CI", "Upper CI"]
            conf_exp = np.exp(conf)

            odds_df = pd.DataFrame({
                "Odds Ratio": odds_ratio,
                "Lower 95% CI": conf_exp["Lower CI"],
                "Upper 95% CI": conf_exp["Upper CI"],
                "p-value": result.pvalues
            })


            # FILTER SIGNIFIKAN
            odds_df_sig = odds_df[odds_df["p-value"] <= alpha]

            if odds_df_sig.empty:
                st.warning("⚠️ Tidak terdapat variabel yang signifikan pada α = 0.05")
            else:
                st.success(f"✅ Variabel signifikan (p-value ≤ {alpha})")
                st.dataframe(
                    odds_df_sig.style.format({
                        "Odds Ratio": "{:.3f}",
                        "Lower 95% CI": "{:.3f}",
                        "Upper 95% CI": "{:.3f}",
                        "p-value": "{:.4f}"
                    }),
                    use_container_width=True
                )

            st.markdown("### 🧠 Interpretasi Odds Ratio")

            interpretasi_list = []

            for var, row in odds_df_sig.iterrows():
                or_val = row["Odds Ratio"]
                lower = row["Lower 95% CI"]
                upper = row["Upper 95% CI"]

                if or_val > 1:
                    efek = "meningkatkan"
                else:
                    efek = "menurunkan"

                interpretasi = (
                    f"Variabel **{var}** memiliki Odds Ratio sebesar **{or_val:.3f}**, "
                    f"yang berarti bahwa penderita **{var}**  memiliki odds sebesar **{or_val:.3f}** kali dalam {efek} kejadian batu empedu."
                )

                interpretasi_list.append(interpretasi)

            for text in interpretasi_list:
                st.write("- " + text)

            
            # ==========================================
            # 4. Penulisan Model dan Interpretasi Koefisien
            # ==========================================
            st.subheader("4️⃣ Penulisan Model dan Interpretasi Koefisien")

            beta_0 = result.params["const"]

            terms = []

            for var, coef in signif_df["Koefisien"].items():
                var_short = var[:2]   # ambil 2 karakter pertama
                terms.append(f"{coef:.4f}{var_short}")

            model_formula = " + ".join(terms)

            st.markdown(rf"""
            Model regresi logistik yang terbentuk adalah sebagai berikut:

            $$
            \hat{{p}} =
            \frac{{e^{{{beta_0:.4f} + {model_formula}}}}}
            {{1 + e^{{{beta_0:.4f} + {model_formula}}}}}
            $$

            dengan:
            - $\hat{{p}}$ : probabilitas kejadian batu empedu
            - $\beta_0$ : konstanta (intersep)
            - Variabel prediktor merupakan variabel yang signifikan secara statistik
            """)
            st.markdown("### 📌 Interpretasi Koefisien Regresi")

            for var, coef in signif_df["Koefisien"].items():
                if coef > 0:
                    arah = "meningkatkan"
                else:
                    arah = "menurunkan"

                st.write(
                    f"- Koefisien variabel **{var}** bernilai **{coef:.4f}**, "
                    f"yang menunjukkan bahwa peningkatan variabel tersebut "
                    f"dapat {arah} peluang terjadinya batu empedu, "
                    f"dengan asumsi variabel lain konstan.")    


    
    def tidak_menggunakan_prepro():
            # =========================
            # Load Data
            # =========================
            df = pd.read_excel("data kesehatan2.xlsx")

            st.subheader("Dataset yang Digunakan")
            st.dataframe(df.head())

            # =========================
            # Target & Prediktor
            # =========================
            y = df["Y_Gallstone Status"]
            X = df.drop(columns=["Y_Gallstone Status"])

            # Ambil variabel numerik saja
            X = X.select_dtypes(include=["int64", "float64"])

            test_size = st.slider(
            "Proporsi data uji:", 0.1, 0.5, 0.3,key="test_size_non_prepro")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=y
            )

            # Tambah konstanta (intercept)
            X_const = sm.add_constant(X_train)

            # =========================
            # Fit Model Logistik
            # =========================
            logit_model = sm.Logit(y_train, X_const)
            result = logit_model.fit(disp=False)

            # =========================
            # 1️⃣ UJI SIMULTAN (Likelihood Ratio Test)
            # =========================
            st.subheader("1️⃣ Uji Signifikansi Simultan (Likelihood Ratio Test)")

            ll_null = result.llnull
            ll_model = result.llf

            LR_stat = 2 * (ll_model - ll_null)
            df_lr = result.df_model
            p_value_lr = chi2.sf(LR_stat, df_lr)

            st.write(f"Nilai Likelihood Ratio (LR) = **{LR_stat:.4f}**")
            st.write(f"Derajat bebas = **{df_lr}**")
            st.write(f"p-value = **{p_value_lr:.4f}**")

            if p_value_lr < 0.05:
                st.success(
                    "Kesimpulan: Model regresi logistik **signifikan secara simultan**, "
                    "artinya seluruh variabel prediktor secara bersama-sama "
                    "berpengaruh terhadap status penyakit batu empedu."
                )
            else:
                st.warning(
                    "Kesimpulan: Model regresi logistik **tidak signifikan secara simultan**."
                )

            # =========================
            # 2️⃣ UJI PARSIAL (WALD TEST)
            # =========================
            st.subheader("2️⃣ Uji Signifikansi Parsial (Wald Test)")

            summary_df = pd.DataFrame({
                "Koefisien": result.params,
                "Std Error": result.bse,
                "Z-Stat": result.tvalues,
                "p-value": result.pvalues
            })
            
            # =========================
            # FILTER SIGNIFIKAN
            # =========================
            alpha = 0.05
            signif_df = summary_df[summary_df["p-value"] < alpha]

            if signif_df.empty:
                st.warning("⚠️ Tidak terdapat variabel yang signifikan secara parsial (p-value < 0,05).")
            else:
                st.markdown("### 🔎 Variabel yang Signifikan secara Parsial (p-value < 0,05)")
                st.dataframe(
                    signif_df.style.format({
                        "Koefisien": "{:.4f}",
                        "Std Error": "{:.4f}",
                        "Z-Stat": "{:.4f}",
                        "p-value": "{:.4f}"
                    }),
                    use_container_width=True
                )

            st.write(
                """
                **Kriteria pengujian**:
                - H₀: βᵢ = 0 (variabel tidak berpengaruh)
                - H₁: βᵢ ≠ 0 (variabel berpengaruh)
                - Tolak H₀ jika p-value < 0,05
                """
            )

            # =========================
            # 3️⃣ ODDS RATIO (SIGNIFIKAN SAJA)
            # =========================
            st.subheader("3️⃣ Odds Ratio (α = 0.05)")

            alpha = 0.05

            odds_ratio = np.exp(result.params)
            conf = result.conf_int()
            conf.columns = ["Lower CI", "Upper CI"]
            conf_exp = np.exp(conf)

            odds_df = pd.DataFrame({
                "Odds Ratio": odds_ratio,
                "Lower 95% CI": conf_exp["Lower CI"],
                "Upper 95% CI": conf_exp["Upper CI"],
                "p-value": result.pvalues
            })


            # FILTER SIGNIFIKAN
            odds_df_sig = odds_df[odds_df["p-value"] <= alpha]

            if odds_df_sig.empty:
                st.warning("⚠️ Tidak terdapat variabel yang signifikan pada α = 0.05")
            else:
                st.success(f"✅ Variabel signifikan (p-value ≤ {alpha})")
                st.dataframe(
                    odds_df_sig.style.format({
                        "Odds Ratio": "{:.3f}",
                        "Lower 95% CI": "{:.3f}",
                        "Upper 95% CI": "{:.3f}",
                        "p-value": "{:.4f}"
                    }),
                    use_container_width=True
                )

            st.markdown("### 🧠 Interpretasi Odds Ratio")

            interpretasi_list = []

            for var, row in odds_df_sig.iterrows():
                or_val = row["Odds Ratio"]
                lower = row["Lower 95% CI"]
                upper = row["Upper 95% CI"]

                if or_val > 1:
                    efek = "meningkatkan"
                else:
                    efek = "menurunkan"

                interpretasi = (
                    f"Variabel **{var}** memiliki Odds Ratio sebesar **{or_val:.3f}**, "
                    f"yang berarti bahwa penderita **{var}**  memiliki odds sebesar **{or_val:.3f}** kali dalam {efek} kejadian batu empedu."
                )

                interpretasi_list.append(interpretasi)

            for text in interpretasi_list:
                st.write("- " + text)

            
            # ==========================================
            # 4. Penulisan Model dan Interpretasi Koefisien
            # ==========================================
            st.subheader("4️⃣ Penulisan Model dan Interpretasi Koefisien")

            beta_0 = result.params["const"]

            terms = []

            for var, coef in signif_df["Koefisien"].items():
                var_short = var[:2]   # ambil 2 karakter pertama
                terms.append(f"{coef:.4f}{var_short}")

            model_formula = " + ".join(terms)

            st.markdown(rf"""
            Model regresi logistik yang terbentuk adalah sebagai berikut:

            $$
            \hat{{p}} =
            \frac{{e^{{{beta_0:.4f} + {model_formula}}}}}
            {{1 + e^{{{beta_0:.4f} + {model_formula}}}}}
            $$

            dengan:
            - $\hat{{p}}$ : probabilitas kejadian batu empedu
            - $\beta_0$ : konstanta (intersep)
            - Variabel prediktor merupakan variabel yang signifikan secara statistik
            """)
            st.markdown("### 📌 Interpretasi Koefisien Regresi")

            for var, coef in signif_df["Koefisien"].items():
                if coef > 0:
                    arah = "meningkatkan"
                else:
                    arah = "menurunkan"

                st.write(
                    f"- Koefisien variabel **{var}** bernilai **{coef:.4f}**, "
                    f"yang menunjukkan bahwa peningkatan variabel tersebut "
                    f"dapat {arah} peluang terjadinya batu empedu, "
                    f"dengan asumsi variabel lain konstan.")


    # ================= PILIHAN PREPROCESSING =================
    choice = st.radio(
        "Apakah anda milih mempreprocessing data?",
        ("Tidak", "Ya"), key = "pilihan_inferensi"
    )

    if choice == "Ya":
        menggunakan_prepro()
        return
    
    tidak_menggunakan_prepro()          
    
    
# uji_diagnostik()