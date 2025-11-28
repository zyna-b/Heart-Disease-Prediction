<div align="center">

# 🫀 Heart Disease Prediction with Machine Learning

### Comprehensive EDA, Statistical Analysis & ML Classification Models

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg?style=for-the-badge)](https://github.com/zyna-b)
[![Made with Love](https://img.shields.io/badge/Made%20with-❤️-red.svg?style=for-the-badge)](https://github.com/zyna-b)

<p align="center">
  <strong>🔬 A complete end-to-end machine learning project for predicting heart disease using clinical data</strong>
</p>

[📊 View Analysis](#-analysis-methodology) • [🚀 Quick Start](#-getting-started) • [📈 Results](#-machine-learning-models--results) • [🤝 Contributing](#-contributing)

</div>

---

## 📌 Table of Contents
- [Overview](#-project-overview)
- [Features](#-key-features)
- [Dataset](#-dataset-information)
- [Analysis Methodology](#-analysis-methodology)
- [ML Models & Results](#-machine-learning-models--results)
- [Tech Stack](#-technical-stack)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 📊 Project Overview

This project implements a **complete machine learning pipeline** for **heart disease prediction** using clinical patient data. It combines comprehensive **exploratory data analysis (EDA)**, **statistical hypothesis testing**, **feature engineering**, and **multiple ML classification algorithms** to build an accurate predictive model.

> 💡 **Why Heart Disease Prediction?**  
> Heart disease is the leading cause of death globally. Early detection through ML-powered analysis can significantly improve patient outcomes and save lives.

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| 📈 **Complete EDA Pipeline** | In-depth data exploration, visualization, and statistical insights |
| 🧪 **Statistical Feature Selection** | T-tests, Chi-square tests with Cohen's d & Cramer's V effect sizes |
| 🧹 **Data Preprocessing** | Missing value imputation, one-hot encoding, standardization |
| 📊 **Advanced Visualizations** | Correlation heatmaps, distribution plots, categorical analysis |
| 🤖 **5 ML Algorithms** | Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM |
| 💾 **Model Deployment Ready** | Saved pickle files for production deployment |

---

## 📁 Dataset Information

| Property | Details |
|----------|---------|
| **Dataset** | Heart Disease Prediction Dataset |
| **Source** | Clinical heart disease patient records |
| **Samples** | 918 patients |
| **Features** | 11 clinical features + 1 target variable |
| **Target** | Binary classification (0 = No Disease, 1 = Disease) |

### 📋 Feature Descriptions

| Feature | Type | Description |
|---------|------|-------------|
| `Age` | Continuous | Patient age in years |
| `Sex` | Categorical | Gender (M/F) |
| `ChestPainType` | Categorical | ATA, NAP, ASY, TA |
| `RestingBP` | Continuous | Resting blood pressure (mm Hg) |
| `Cholesterol` | Continuous | Serum cholesterol (mg/dl) |
| `FastingBS` | Binary | Fasting blood sugar > 120 mg/dl |
| `RestingECG` | Categorical | Resting ECG results (Normal, ST, LVH) |
| `MaxHR` | Continuous | Maximum heart rate achieved |
| `ExerciseAngina` | Binary | Exercise-induced angina (Y/N) |
| `Oldpeak` | Continuous | ST depression induced by exercise |
| `ST_Slope` | Categorical | Slope of peak exercise ST segment |

---

## 🔬 Analysis Methodology

### 1️⃣ Exploratory Data Analysis (EDA)
- Data structure and quality assessment
- Missing value analysis and treatment (mean imputation for Cholesterol & RestingBP)
- Distribution analysis of continuous variables
- Target variable balance analysis

### 2️⃣ Data Visualization
```
📊 Histograms with KDE → Continuous variable distributions
📊 Count Plots → Categorical variables vs Heart Disease
📊 Box Plots & Violin Plots → Distribution comparisons
📊 Correlation Heatmap → Feature relationships
```

### 3️⃣ Statistical Feature Selection
| Test | Variables | Metrics |
|------|-----------|---------|
| **T-Test** | Continuous (Age, RestingBP, Cholesterol, MaxHR, Oldpeak) | Cohen's d effect size |
| **Chi-Square** | Categorical (Sex, ChestPainType, RestingECG, etc.) | Cramer's V |

- **Significance Level**: α = 0.05
- **Features selected based on p-value and effect size**

### 4️⃣ Data Preprocessing
- ✅ One-hot encoding for categorical variables
- ✅ StandardScaler normalization for continuous features
- ✅ 80/20 train-test split with random state for reproducibility

---

## 🤖 Machine Learning Models & Results

### Models Implemented
| Model | Description |
|-------|-------------|
| **Logistic Regression** | Linear classifier for binary outcomes |
| **K-Nearest Neighbors (KNN)** | Instance-based learning algorithm |
| **Gaussian Naive Bayes** | Probabilistic classifier |
| **Decision Tree** | Tree-based classification |
| **Support Vector Machine (RBF)** | Kernel-based classifier |

### 📈 Performance Metrics
Models are evaluated using:
- **Accuracy Score**
- **F1 Score** (harmonic mean of precision and recall)
- **Classification Report**

### 💾 Model Artifacts
The project saves deployment-ready artifacts:
```python
KNN_heart.pkl    # Trained KNN model
scaler.pkl       # StandardScaler for preprocessing
columns.pkl      # Feature column names
```

---

## 🛠️ Technical Stack

<table>
<tr>
<td>

**Data Science**
- Python 3.9+
- Pandas
- NumPy
- SciPy

</td>
<td>

**Visualization**
- Matplotlib
- Seaborn
- Plotly (optional)

</td>
<td>

**Machine Learning**
- Scikit-learn
- Joblib

</td>
<td>

**Environment**
- Jupyter Notebook
- VS Code

</td>
</tr>
</table>

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.9+
Jupyter Notebook or VS Code
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/zyna-b/heart-disease-ml-prediction.git
cd heart-disease-ml-prediction
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the notebooks**
```bash
# For EDA and Statistical Analysis
jupyter notebook heart_disease_analysis.ipynb

# For ML Model Training
jupyter notebook heart_disease_prediction.ipynb
```

---

## 📁 Project Structure

```
heart-disease-ml-prediction/
│
├── 📓 heart_disease_analysis.ipynb    # EDA & Statistical Analysis
├── 📓 heart_disease_prediction.ipynb  # ML Model Training & Evaluation
├── 📊 heart.csv                       # Dataset
├── 📄 requirements.txt                # Python dependencies
├── 📄 README.md                       # Project documentation
│
└── 📦 Model Artifacts (generated after training)
    ├── KNN_heart.pkl                  # Trained model
    ├── scaler.pkl                     # Feature scaler
    └── columns.pkl                    # Feature columns
```

---

## 🔍 Statistical Analysis Highlights

### Hypothesis Testing Framework
| Component | Description |
|-----------|-------------|
| **H₀ (Null)** | No significant difference between groups |
| **H₁ (Alternative)** | Significant difference exists |
| **α (Significance)** | 0.05 |
| **Effect Sizes** | Cohen's d (continuous), Cramer's V (categorical) |

### Feature Selection Criteria
- ✅ P-value < 0.05 for statistical significance
- ✅ Effect size consideration for practical importance
- ✅ Clinical relevance for medical interpretation

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit changes (`git commit -m 'Add AmazingFeature'`)
4. 📤 Push to branch (`git push origin feature/AmazingFeature`)
5. 🔃 Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🏷️ Keywords

`heart disease prediction` `machine learning` `classification` `healthcare AI` `medical data analysis` `exploratory data analysis` `statistical testing` `feature selection` `scikit-learn` `python data science` `cardiovascular analysis` `predictive healthcare` `clinical decision support` `binary classification` `KNN classifier` `logistic regression` `SVM` `decision tree` `naive bayes`

---

## 📧 Contact

<div align="center">

**👩‍💻 Zainab Hamid**

[![Email](https://img.shields.io/badge/Email-zainabhamid2468%40gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:zainabhamid2468@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/zainab-hamid-187a18321/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/zyna-b)

---

### ⭐ Star this repository if you found it helpful!

### 🔗 Share with others interested in healthcare ML and data science!

</div>
