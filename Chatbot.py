# ================================
# EV Sales Prediction Chatbot
# Fully Robust Version with Historical Plot
# ================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

# -------------------------------
# Step 1: Load dataset
# -------------------------------
file_path = '/kaggle/input/electric-car-sales-2010-2024/IEA-EV-dataEV salesHistoricalCars.csv'
df = pd.read_csv(file_path)

# -------------------------------
# Step 2: Clean dataset
# -------------------------------
df = df.dropna()
df['year'] = df['year'].astype(int)
df['value'] = pd.to_numeric(df['value'], errors='coerce')
df = df.dropna(subset=['value'])

# Drop irrelevant descriptive columns if present
drop_cols = ['parameter', 'country', 'region_code']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Save a copy of original strings for historical plotting
df_original = df.copy()

# -------------------------------
# Step 3: Encode categorical columns
# -------------------------------
cat_cols = df.select_dtypes(include=['object']).columns
le_dict = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le
    print(f"Encoded column: {col}")

# Features and target
X = df.drop(columns=['value'])
y = df['value']

# -------------------------------
# Step 4: Scale features
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Step 5: Train model
# -------------------------------
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_scaled, y)

# -------------------------------
# Step 6: Chatbot function
# -------------------------------
def ev_chatbot():
    print("🤖 Hello! I am EV-SalesBot. I can predict EV sales value and show historical trends.\n")
    
    while True:
        try:
            # Ask for year
            year = int(input("Enter the Year (e.g., 2025): "))
            user_input = {'year': year}
            user_filter = {}  # for historical plot filtering
            
            # Handle categorical inputs
            for col in cat_cols:
                le = le_dict[col]
                classes = list(le.classes_)
                
                print(f"Available options for {col}: {classes}")
                
                while True:
                    val = input(f"Enter {col}: ").strip()
                    val_clean = val.lower()
                    
                    # Match ignoring case and spaces
                    match = None
                    for cls in classes:
                        if cls.strip().lower() == val_clean:
                            match = cls
                            break
                    
                    if match is not None:
                        # Encoded value for prediction
                        user_input[col] = le.transform([match])[0]
                        # Original string for historical plot
                        user_filter[col] = match
                        break
                    else:
                        print(f"⚠️ Invalid option! Please choose from: {classes}")
            
            # Convert to DataFrame and scale
            user_df = pd.DataFrame([user_input])
            user_df = user_df[X.columns]  # ensure correct column order
            user_scaled = scaler.transform(user_df)
            
            # Predict
            prediction = model.predict(user_scaled)[0]
            print(f"\n🤖 Predicted EV Sales Value for {year}: {prediction:.2f}\n")
            
            # -------------------------------
            # Plot historical data
            # -------------------------------
            plot_df = df_original.copy()
            for col, val in user_filter.items():
                # Robust filtering using str.contains (ignores case and extra spaces)
                plot_df = plot_df[plot_df[col].str.strip().str.contains(val.strip(), case=False, regex=False)]
            
            if plot_df.empty:
                print("⚠️ No historical data available for the selected filters.\n")
            else:
                plt.figure(figsize=(10,6))
                sns.lineplot(data=plot_df.sort_values('year'), x='year', y='value', marker='o')
                plt.title("Historical EV Sales Trend for Selected Filters")
                plt.xlabel("Year")
                plt.ylabel("EV Sales Value")
                plt.show()
        
        except ValueError:
            print("⚠️ Invalid input! Please enter a valid number.\n")
        except Exception as e:
            print(f"⚠️ Error: {e}\n")
        
        cont = input("Do you want to predict another value? (yes/no): ").strip().lower()
        if cont != 'yes':
            print("🤖 Goodbye! Keep following EV trends 🚗⚡")
            break

# -------------------------------
# Step 7: Run chatbot
# -------------------------------
if __name__ == "__main__":
    ev_chatbot()
