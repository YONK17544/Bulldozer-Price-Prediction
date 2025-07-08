# 🚜 Bulldozer Price Prediction (Blue Book for Bulldozers)

This project builds a machine learning model that predicts the final sale price of bulldozers based on their attributes and auction data. It is based on the Kaggle competition "Blue Book for Bulldozers" and demonstrates the full machine learning pipeline from data cleaning to model evaluation.

---

## 📊 Overview

- **Dataset:** [Blue Book for Bulldozers](https://www.kaggle.com/competitions/bluebook-for-bulldozers)
- **Goal:** Predict the `SalePrice` of a bulldozer at auction
- **Rows:** ~412,000+
- **Columns:** ~50+
- **Target Variable:** `SalePrice` (continuous, regression task)

---

## 🧠 What This Project Covers

- Exploratory Data Analysis (EDA)
- Handling missing data (numeric & categorical)
- Feature engineering:
  - Date parsing (year, month, day)
  - Binary missing indicators
  - Categorical encoding with category codes
- Model training using:
  - `RandomForestRegressor`
- Hyperparameter tuning with:
  - `RandomizedSearchCV`
- Evaluation using:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Log Error (RMSLE)
  - R² Score

---

## 📈 Model Results

Final tuned model performance:

Training MAE: ~2958
Validation MAE: ~5965

Training RMSLE: ~0.144
Validation RMSLE: ~0.245

Training R²: ~0.958
Validation R²: ~0.881

These scores show that the model generalizes well and captures patterns in both the training and validation sets without overfitting.

---

## 🧰 Tools & Libraries Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 📁 Project Structure
bulldozer_price_prediction/
├── data/ # Raw and processed data files
├── notebooks/ # Jupyter Notebooks for EDA and modeling
├── models/ # Saved model files
├── scripts/ # Utility scripts for preprocessing and evaluation
└── README.md # Project description


---

## 🚀 Next Steps

- Try advanced models: XGBoost, LightGBM, or CatBoost
- Train deep learning models (e.g., with PyTorch or TensorFlow)
- Deploy as a web app using Flask or FastAPI
- Monitor model performance over time with real-world input

---

## 📌 Credits

- Kaggle & Blue Book for Bulldozers competition
- Dataset originally provided by Sandhills Global

---


