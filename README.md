# Predictive_Maintenance_System
# Predictive Maintenance - Machine Failure Classification 🔧🤖

This project uses machine learning to predict whether a machine is likely to fail or not, based on sensor data like temperature, torque, speed, etc. The goal is to help schedule maintenance in advance to avoid unexpected breakdowns.

## 🗂 Dataset

I used the [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) from Kaggle.  
It contains sensor readings of manufacturing machines and indicates whether a failure occurred.

- Features include:
  - Air temperature [K]
  - Process temperature [K]
  - Rotational speed [rpm]
  - Torque [Nm]
  - Tool wear [min]
  - Type of machine (categorical)
- Target variable:
  - `Machine failure` (0 or 1)

## 🧪 ML Approach

1. **Data Preprocessing:**
   - Dropped columns like `UDI` and `Product ID` (not useful for modeling).
   - Renamed the target column to `Failure` for simplicity.
   - One-hot encoded the machine type.
   - Standardized features using `StandardScaler`.

2. **Modeling:**
   - Trained two models:
     - Random Forest
     - XGBoost
   - Compared their performance on accuracy and classification report.

3. **Hyperparameter Tuning:**
   - Used `GridSearchCV` to find the best parameters for XGBoost.
   - Evaluated final performance using the best XGBoost model.

4. **Model Saving:**
   - Exported the trained model and scaler using `joblib`.

5. **Prediction Function:**
   - Wrote a function that takes new machine data and returns failure probability.
   - Also gives a simple recommendation (e.g., schedule maintenance or not).

## 📈 Results

- Both models performed well, but XGBoost gave slightly better accuracy.
- Feature importance showed that tool wear, torque, and temperature were the most important.

## 💡 Example Prediction

I tested the model on a sample machine with the following values:

```python
sample_data = pd.DataFrame({
    'Type': ['L'],
    'Air temperature [K]': [300],
    'Process temperature [K]': [310],
    'Rotational speed [rpm]': [1500],
    'Torque [Nm]': [40],
    'Tool wear [min]': [200]
})
