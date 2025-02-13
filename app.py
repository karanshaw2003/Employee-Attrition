from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained models
with open("logistic_regression_model.pkl", "rb") as log_model_file:
    log_reg = pickle.load(log_model_file)

with open("random_forest_model.pkl", "rb") as rf_model_file:
    rf_clf = pickle.load(rf_model_file)

@app.route('/')
def home():
    return render_template('index.html', prediction_result=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        model_type = data.get("model", "random_forest")
        
        # Extract input features
        features = [
            int(data["age"]),
            int(data["job_level"]),
            int(data["monthly_income"]),
            int(data["years_at_company"]),
            int(data["distance_from_home"]),
            int(data["num_companies_worked"]),
            int(data["percent_salary_hike"]),
            int(data["training_times_last_year"]),
            int(data["years_since_last_promotion"]),
            int(data["years_with_curr_manager"])
        ]
        
        # Define the feature column names as per the trained model
        feature_names = [
            "Age", "JobLevel", "MonthlyIncome", "YearsAtCompany", "DistanceFromHome",
            "NumCompaniesWorked", "PercentSalaryHike", "TrainingTimesLastYear",
            "YearsSinceLastPromotion", "YearsWithCurrManager"
        ]
        input_data = pd.DataFrame([features], columns=feature_names)
        
        # Make prediction
        if model_type == "logistic_regression":
            prediction = log_reg.predict(input_data)[0]
        else:
            prediction = rf_clf.predict(input_data)[0]
        
        result = "Yes (Attrition)" if prediction == 1 else "No (Stay)"
        return render_template('index.html', prediction_result=result)
    except Exception as e:
        return render_template('index.html', prediction_result=f"Error: {str(e)}")
   
if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=10000)


