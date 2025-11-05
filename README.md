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

├── Data/
│ ├── Predicted_Data/
│ │ ├── Bits main gate/
│ │ ├── Bus stand/
│ │ ├── Dust Incense/
│ │ ├── Evening walk/
│ │ ├── Good Night/
│ │ ├── High_Conc/
│ │ ├── KITCHEN/
│ │ ├── LOW_CONC/
│ │ └── Main_gate/
│ │
│ └── Raw Data/
│ ├── Bits_Main_Gate.xlsx
│ ├── Bus Stand.xlsx
│ ├── Dust$Incense.xlsx
│ ├── Evening walk and Good night.xlsx
│ ├── Good night experiment.xlsx
│ ├── High_Conc_Exp.xlsx
│ ├── Honeywell_01_prediction copy.xlsx
│ └── Honeywell_01_prediction.xlsx
│
├── Notebooks/
│ ├── Calibration_1.ipynb
│ ├── Calibration_2.ipynb
│ ├── Calibration_3.py
│ ├── Calibration_4.py
│ ├── Calibration_5.ipynb
│ ├── Calibration_6.ipynb
│ └── Calibration_7.ipynb
│
├── INCOTHERM 2025_submission_63_paper_v2.pdf
├── requirements.txt
├── .gitattributes
└── README.md

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


