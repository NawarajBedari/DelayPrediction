from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import holidays
import random
import json
import os

# --- Load model and encoders ---
with open("model/delay_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/target_encoder.pkl", "rb") as f:
    target_encoder = pickle.load(f)

with open("model/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# --- Load original dataset to get station names ---
df_raw = pd.read_csv(r"C:\Users\Asus\Desktop\AIDI2ndSemester\AI Project\Final\processed_delay_data.csv")
unique_stations = sorted(df_raw["Station"].unique().tolist())

# --- Load coordinates dictionary ---
coord_path = r"C:\Users\Asus\DelayPrediction\station_coords.json"
if os.path.exists(coord_path):
    with open(coord_path, "r") as f:
        station_coords = json.load(f)
else:
    station_coords = {}

# --- Setup ---
template_path = r"C:\Users\Asus\DelayPrediction\Templates"
app = Flask(__name__, template_folder=template_path)
can_holidays = holidays.CA(prov='ON')  # Ontario holidays

# --- Rule-based override ---
def apply_custom_rules(row):
    month = row["Month"]
    hour = row["Hour"]
    is_weekend = row["Is_Weekend"]
    is_peak = row["Is_Peak_Hour"]
    date_obj = datetime.strptime(row["Date"], "%Y-%m-%d")

    if date_obj in can_holidays:
        return random.choice(["Short", "Medium", "Long"])
    if month in [12, 1, 2]:
        return random.choices(["Medium", "Long"], weights=[0.6, 0.4])[0]
    if is_weekend:
        return random.choices(["Short", "Medium"], weights=[0.7, 0.3])[0]
    if is_peak == 1:
        return random.choice(["Short", "Medium", "Long"])
    return None

# --- Web Form Route ---
@app.route('/', methods=['GET'])
def form():
    return render_template("form.html", stations=unique_stations)

# --- Prediction Form POST handler ---
@app.route('/predict', methods=['POST'])
def predict():
    date_str = request.form.get("date")
    time_str = request.form.get("time")
    station_name = request.form.get("station")

    # Parse time and generate future hours
    input_time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    hours = list(range(input_time.hour, 24))

    # Build input records
    records = []
    for hour in hours:
        row = {
            "Date": date_str,
            "Hour": hour,
            "Station": station_name,
            "Day": input_time.strftime("%A"),
            "Month": input_time.month,
            "Is_Weekend": input_time.weekday() >= 5,
            "Is_Peak_Hour": int(hour in list(range(7, 10)) + list(range(16, 19))),
            "Time_Category": (
                "Morning" if hour < 12 else
                "Afternoon" if hour < 17 else
                "Evening"
            ),
            "Code": "UNKNOWN",
            "Bound": "N",
            "Line": "YU",
            "Vehicle": 0,
            "Min Gap": 0,
        }
        records.append(row)

    # Create DataFrame
    df_input = pd.DataFrame(records)

    # Encode all features
    for col in ['Day', 'Station', 'Code', 'Bound', 'Line', 'Time_Category']:
        if col in label_encoders:
            le = label_encoders[col]
            df_input[col] = df_input[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df_input[col] = le.transform(df_input[col])

    # Predict using model
    expected_order = ['Day', 'Station', 'Code', 'Min Gap', 'Bound', 'Line', 'Vehicle',
                      'Hour', 'Time_Category', 'Is_Weekend', 'Month', 'Is_Peak_Hour']
    features = df_input[expected_order]
    model_preds = model.predict(features)
    decoded_preds = target_encoder.inverse_transform(model_preds)

    # Apply rule-based overrides
    final_preds = []
    for i, row in df_input.iterrows():
        rule_based = apply_custom_rules(records[i])
        final_preds.append(rule_based if rule_based else decoded_preds[i])

    # Add predictions to DataFrame
    df_input["Date"] = [r["Date"] for r in records]
    df_input["Hour"] = [r["Hour"] for r in records]
    df_input["Station"] = [r["Station"] for r in records]
    df_input["Delay_Category"] = final_preds

    # Handle predictions CSV file with data control
    output_path = "delay_predictions.csv"
    predictions_to_save = df_input[["Date", "Hour", "Station", "Delay_Category"]]
    
    # If file exists, load it and combine with new predictions
    if os.path.exists(output_path):
        try:
            existing_predictions = pd.read_csv(output_path)
            # Combine old and new predictions
            combined_predictions = pd.concat([existing_predictions, predictions_to_save])
            # Remove duplicates based on all columns
            combined_predictions = combined_predictions.drop_duplicates()
            # Sort by Date and Hour for better organization
            combined_predictions = combined_predictions.sort_values(by=["Date", "Hour"])
            # Save the complete dataset
            combined_predictions.to_csv(output_path, index=False)
        except Exception as e:
            # If there's an error reading existing file, just save new predictions
            print(f"Error reading existing predictions: {e}")
            predictions_to_save.to_csv(output_path, index=False)
    else:
        # If file doesn't exist, create it with current predictions
        predictions_to_save.to_csv(output_path, index=False)

    # Visualization Data
    prediction_pairs = list(zip(df_input["Hour"], df_input["Delay_Category"]))
    chart_data = [{'hour': f"{hour}:00", 'delay': delay} for hour, delay in prediction_pairs]
    selected_coords = station_coords.get(station_name, [43.6532, -79.3832])  # fallback to Toronto downtown

    return render_template("form.html",
                         predictions=prediction_pairs,
                         date=date_str,
                         station=station_name,
                         stations=unique_stations,
                         chart_data=json.dumps(chart_data),
                         station_coords=selected_coords)

if __name__ == '__main__':
    app.run(debug=True)