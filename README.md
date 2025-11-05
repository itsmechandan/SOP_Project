# 🧪 Performance Evaluation of Low-Cost Sensors Under Different Aerosols

**Authors:** Deepali Agrawal¹, Jakka Venkat Chandan¹, Anil Kumar Saini², Aakash C. Rai³, and Prateek Kala¹  
**Affiliations:**  
¹ Department of Mechanical Engineering, Birla Institute of Technology and Science (BITS) Pilani, India  
² Senior Scientist, SEG Design Group, CSIR–CEERI Pilani, India  
³ Department of Sustainable Energy Engineering, Indian Institute of Technology Kanpur, India  

---

## 🏷️ Conference
**1st International Conference on Thermofluids Engineering (INCOTHERM 2025)**  
*IIT (ISM) Dhanbad, October 10th–11th, 2025*  
**Paper ID:** INCOTHERM2025–063

---

## 📖 Overview

This repository contains the **machine learning calibration work** for the research paper  
**“Performance Evaluation of Low-Cost Sensors Under Different Aerosols.”**

The project focuses on **calibrating low-cost particulate matter (PM) sensors** using advanced regression algorithms.  
Low-cost sensors (LCSs) are widely used for air quality monitoring but often suffer from inaccuracies due to temperature, humidity, and aerosol composition.  
By leveraging **machine learning models**, we aim to improve their performance relative to a **high-end reference instrument (GRIMM 11-A)**.

---

## 🌫️ Research Summary

The study evaluates the performance of low-cost sensors under **four aerosol conditions**:

- Arizona road dust  
- Compressor oil  
- Incense smoke  
- Sodium chloride (NaCl) particles  

Each sensor was exposed to controlled aerosols in an environmental chamber, and their raw outputs were compared to the **GRIMM 11-A reference sensor**.  
Calibration was performed using several algorithms, including both **statistical** and **machine learning-based** approaches.

---

## ⚙️ Sensors Used

| Category | Model | Description |
|-----------|--------|-------------|
| Low-cost | Plantower PMSA003 | Optical particle counter (PM1, PM2.5, PM10) |
| Low-cost | Sensirion SPS30 | Laser-based PM sensor with temperature and humidity compensation |
| Reference | GRIMM 11-A | High-accuracy optical particle counter used as the calibration reference |

---

## 🧠 Methodology

1. **Data Collection**
   - Parallel measurement using Plantower and Sensirion sensors alongside GRIMM 11-A.
   - Experiments conducted under controlled aerosol conditions.

2. **Data Preprocessing**
   - Cleaning, synchronization, and filtering of time-series data.  
   - Normalization and feature selection for ML models.

3. **Machine Learning Calibration**
   - Algorithms used:
     - Random Forest Regressor (RFR)
     - Support Vector Regressor (SVR)
     - XGBoost Regressor
     - Artificial Neural Network (for reference comparison)
     - Linear & Quadratic Regression (baseline)

4. **Evaluation Metrics**
   - Coefficient of Determination (**R²**)  
   - Root Mean Square Error (**RMSE**)  
   - Mean Absolute Error (**MAE**)

---

## 📊 Key Observations

- **Machine learning models** significantly outperformed linear and quadratic calibration.  
- **Sensirion SPS30** showed the best agreement with GRIMM 11-A, with RMSE values < 7 µg/m³.  
- **Random Forest** and **XGBoost** achieved the highest R² and lowest RMSE across all aerosol types.  
- The calibration successfully reduced bias and improved overall sensor reliability.

*(Sample visualizations and plots can be found in the notebooks.)*

---

## 📁 Repository Structure


### 🔍 Current Files in This Repository

| Old File / Folder Name                      | New Path / Name                                   |
| ------------------------------------------- | ------------------------------------------------- |
| `Calibration_1.ipynb`                       | `Notebooks/01_random_forest_calibration.ipynb`    |
| `Calibration_2.ipynb`                       | `Notebooks/02_linear_quadratic_calibration.ipynb` |
| `Calibration_3.py`                          | `Notebooks/03_svr_calibration.py`                 |
| `Calibration_4.py`                          | `Notebooks/04_xgboost_calibration.py`             |
| `Calibration_5.ipynb`                       | `Notebooks/05_ann_calibration.ipynb`              |
| `Calibration_6.ipynb`                       | `Notebooks/06_combined_calibration.ipynb`         |
| `Calibration_7.ipynb`                       | `Notebooks/07_final_model_comparison.ipynb`       |
| `Data/Raw Data/`                            | `Data/Raw_Data/` (rename for consistency)         |
| `Data/Predicted_Data/Bits main gate`        | `Data/Predicted_Data/Bits_Main_Gate/`             |
| `Data/Predicted_Data/Bus stand`             | `Data/Predicted_Data/Bus_Stand/`                  |
| `Data/Predicted_Data/Dust Incense`          | `Data/Predicted_Data/Dust_Incense/`               |
| `Data/Predicted_Data/Evening walk`          | `Data/Predicted_Data/Evening_Walk/`               |
| `Data/Predicted_Data/Good Night`            | `Data/Predicted_Data/Good_Night/`                 |
| `Data/Predicted_Data/High_Conc`             | (keep same)                                       |
| `Data/Predicted_Data/KITCHEN`               | `Data/Predicted_Data/Kitchen/`                    |
| `Data/Predicted_Data/LOW_CONC`              | `Data/Predicted_Data/Low_Conc/`                   |
| `Data/Predicted_Data/Main_gate`             | `Data/Predicted_Data/Main_Gate/`                  |
| `INCOTHERM 2025_submission_63_paper_v2.pdf` | `Reports/INCOTHERM_2025_Paper.pdf`                |
| `.gitignore` *(optional, create new)*       | `.gitignore`                                      |
| `.gitattributes`                            | (remove — not required unless needed)             |
| `README.md`                                 | (keep in main folder)                             |
| `requirements.txt` *(new)*                  | (keep in main folder)                             |
| `venv/` *(new virtual environment)*         | (auto-created locally, ignored in GitHub)         |


## 📈 Sample Results (Placeholder)

| Model | R² | RMSE (µg/m³) | MAE (µg/m³) |
|--------|----|---------------|--------------|
| Random Forest | 0.93 | 2.1 | 1.5 |
| SVR | 0.90 | 2.4 | 1.7 |
| XGBoost | 0.94 | 2.0 | 1.4 |

*(Actual plots and metrics are in the Jupyter notebooks.)*

---



## 🧑‍💻 Author

**Jakka Venkat Chandan**  
Under the guidance of **Dr. Prateek Kala**  
Department of Mechanical Engineering,  
Birla Institute of Technology and Science (BITS) Pilani, India  

📧 *jakkavenkatchandan@gmail.com*  
📄 [[LinkedIn Profile /] ](https://www.linkedin.com/in/jakka-venkat-chandan-4269b124a/)

---


---

## 🧰 Tech Stack

- **Python** (NumPy, Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn)
- **Jupyter Notebooks**
- **Git/GitHub** for version control
- **Excel** for experimental data management
- **Different Types of Sensors**

---


