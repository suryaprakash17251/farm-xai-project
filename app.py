# 🌾 Farm AI — Explainable Agronomy Recommendation System

An end-to-end AI system that recommends farming actions (irrigate, fertilize, apply pesticide, harvest, or no action) based on soil, weather, and crop sensor data — with full SHAP explainability.

Built as a portfolio project for Jacob AI's Fresh Engineer role.

---

## 🎯 What It Does

| Page | Description |
|------|-------------|
| 🏠 Overview | Dataset summary, action distribution, model accuracy |
| 🔮 Get Recommendation | Input your field conditions → get AI recommendation with confidence scores |
| 🧠 Explain Prediction | SHAP waterfall chart showing WHY the AI made its decision |
| 📊 Model Insights | Global feature importance, per-class SHAP, correlation heatmap |
| 📋 Dataset Explorer | Filter, browse, and download the training data |

---

## 🏗️ Architecture

```
farm_xai/
├── app.py                  ← Streamlit entry point
├── requirements.txt
├── data/
│   └── generate_data.py    ← Synthetic farm dataset generator
├── models/                 ← Saved model artifacts (auto-created)
├── utils/
│   └── model_utils.py      ← XGBoost training + SHAP utilities
└── pages/
    ├── overview.py         ← Overview dashboard
    ├── recommendation.py   ← Recommendation input form
    ├── explanation.py      ← SHAP explanation visualizations
    ├── insights.py         ← Global model explainability
    └── explorer.py         ← Dataset browser
```

---

## ⚡ Quick Start

### 1. Clone / download this project

```bash
git clone <your-repo>
cd farm_xai
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

> **Note:** On first launch, the model trains automatically (~10 seconds). Subsequent loads use the cached model.

---

## 🤖 Model Details

- **Algorithm:** XGBoost (Gradient Boosted Trees)
- **Task:** Multi-class classification (5 actions)
- **Features:** 18 (15 numeric + 3 categorical)
- **Training data:** 2,000 synthetic farm records
- **Accuracy:** ~90%+ on test set
- **Explainability:** SHAP TreeExplainer (both global + local)

### Features Used

**Soil:** moisture, pH, nitrogen, phosphorus, potassium, organic matter  
**Weather:** temperature, 7-day rainfall, humidity, solar radiation, wind speed  
**Crop:** age, leaf area index, NDVI score, pest pressure index  
**Categorical:** crop type, soil type, region  

### Recommended Actions

| Class | Action | Trigger Condition |
|-------|--------|-------------------|
| 0 | ✅ No Action | Conditions are optimal |
| 1 | 💧 Irrigate | Low soil moisture + low rainfall |
| 2 | 🌱 Fertilize | Low nitrogen or phosphorus |
| 3 | 🐛 Apply Pesticide | High pest pressure index |
| 4 | 🌾 Harvest | Mature crop + high NDVI |

---

## 🧠 Explainability Approach

This project uses **SHAP (SHapley Additive exPlanations)** — the gold standard for ML explainability:

- **Local explanations:** For each prediction, a waterfall chart shows how much each feature pushed the model toward or away from the predicted action.
- **Global explanations:** Mean |SHAP| values across all records reveal which features are most important overall and per action class.
- **Plain-language summary:** Auto-generated text explains the top 2 drivers in simple terms.

This approach is directly aligned with Jacob AI's mission of **explainable AI for agronomy**.

---

## 📦 Extending This Project

Ideas to level it up further:

- Connect real sensor APIs (IoT farm data)
- Add a map visualization using GeoPandas/Folium
- Replace synthetic data with real datasets (e.g., USDA, FAO)
- Add a REST API using FastAPI for mobile/web integration
- Deploy on AWS/GCP with Docker

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — interactive web dashboard
- **XGBoost** — gradient boosted tree model
- **SHAP** — model explainability
- **Scikit-learn** — preprocessing, evaluation
- **Plotly** — interactive charts
- **Pandas / NumPy** — data handling

---

*Built for Jacob AI's Agri-Tech R&D team. Demonstrates explainable AI for agronomic decision-making.*
