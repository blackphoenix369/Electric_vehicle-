# Electric_vehicle-

# ⚡ Electric Vehicle (EV) Market Analysis & Regression (2010–2024)

This project analyzes global Electric Vehicle (EV) market data from 2010 to 2024, focusing on sales, stock, and market share across multiple regions and vehicle types.
The objective is to understand EV adoption trends and build regression models to predict future sales and market behavior.

---

## 📊 Project Overview

The dataset provides insights into the growth of electric mobility around the world.
Each record includes data on year, region, vehicle mode, powertrain type, and value metrics.

The workflow includes:

* Data cleaning and preprocessing
* Exploratory Data Analysis (EDA)
* Feature encoding and scaling
* Regression model training and evaluation
* Visualization of actual vs predicted trends

---

## 🗂️ Dataset Details

**Source:** Electric Vehicle Market Dataset (2010–2024)

**Key Columns:**

* `year` – Year of observation
* `region` – Geographic region or country
* `mode` – Vehicle type (e.g., cars, buses)
* `powertrain` – Powertrain category (BEV, PHEV, etc.)
* `unit` – Measurement unit (e.g., sales, stock)
* `value` – Numeric value representing sales or share

---

## 🧠 Machine Learning Objective

To build regression models capable of accurately predicting **EV sales values** based on year and categorical features like region, mode, and powertrain.
The project experiments with multiple regression techniques to improve prediction accuracy.

---

## 🧩 Dependencies

Install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

---

## 🧱 Project Structure

```
EV-Market-Analysis/
│
├── data/
│   └── IEA-EV-dataEV-salesHistoricalCars.csv
│
├── EV_Market_Regression.ipynb    # Main analysis notebook/script
├── README.md                     # Project description
└── app.py.                       # code
```

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/EV-Market-Analysis.git
   cd EV-Market-Analysis
   ```

2. Open the notebook in Jupyter or VS Code:

   ```bash
   jupyter notebook EV_Market_Regression.ipynb
   ```

3. Run all cells step by step.

---

## 📈 Project Workflow

1. **Data Cleaning**

   * Removal of missing and duplicate values
   * Type conversions for numeric fields

2. **Exploratory Data Analysis**

   * Visualizations for year-wise sales trends
   * Region and category distribution plots

3. **Feature Encoding and Scaling**

   * Label encoding for categorical columns
   * Standard scaling for numerical features

4. **Model Training**

   * Linear Regression as baseline
   * Random Forest and XGBoost for improved accuracy

5. **Model Evaluation**

   * R² score
   * Mean Squared Error (MSE)
   * Mean Absolute Error (MAE)

6. **Visualization**

   * Actual vs Predicted plots
   * Yearly growth trend graphs

---

## 📊 Results Summary

The project demonstrates that tree-based ensemble models such as **Random Forest** and **XGBoost** outperform simple linear regression.
XGBoost achieves the highest accuracy and best generalization on unseen data.

---

## 💡 Future Enhancements

* Incorporate **time-series forecasting** (ARIMA, Prophet, or LSTM)
* Include **EV charging infrastructure data** for richer insights
* Build an **interactive dashboard** using Streamlit or Plotly

---

