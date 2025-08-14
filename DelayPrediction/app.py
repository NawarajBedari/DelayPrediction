from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import holidays
import random
import requests
import json
import os

# === Setup ===
template_path = r"C:\Users\Asus\Documents\FlaskTemplates"
app = Flask(__name__, template_folder=template_path)
can_holidays = holidays.CA(prov='ON')
WEATHER_API_KEY = "P2P9RT5GW6NZWJX4YH9QPFBND"
WEATHER_LOCATION = "Toronto,ON"

# === Load model and encoders ===
with open("model/delay_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/target_encoder.pkl", "rb") as f:
    target_encoder = pickle.load(f)
with open("model/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# === Load station list & coordinates ===
df_raw = pd.read_csv(r"C:\Users\Asus\Desktop\AIDI2ndSemester\AI Project\Final\processed_delay_data.csv")
unique_stations = sorted(df_raw["Station"].unique().tolist())

coord_path = r"C:\Users\Asus\DelayPrediction\station_coords.json"
if os.path.exists(coord_path):
    with open(coord_path, "r") as f:
        station_coord_dict = json.load(f)
else:
    station_coord_dict = {}

# === Fetch Weather ===
def fetch_weather(date):
    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        f"{WEATHER_LOCATION}/{date}?unitGroup=metric&key={WEATHER_API_KEY}&include=days&contentType=json"
    )
    try:
        res = requests.get(url)
        data = res.json()
        day = data['days'][0]
        return {
            "temp": day.get('temp', "N/A"),
            "conditions": day.get('conditions', "Unavailable"),
            "precip": day.get('precip', 0),
            "snow": day.get('snow', 0)
        }
    except:
        return {"temp": "N/A", "conditions": "Unavailable", "precip": 0, "snow": 0}

# === Rule-based override ===
def apply_custom_rules(row, weather):
    month = row["Month"]
    hour = row["Hour"]
    is_weekend = row["Is_Weekend"]
    is_peak = row["Is_Peak_Hour"]
    date_obj = datetime.strptime(row["Date"], "%Y-%m-%d")

    snow_val = weather.get("snow") or 0
    precip_val = weather.get("precip") or 0

    if date_obj in can_holidays:
        return random.choice(["Short", "Medium", "Long"])
    if snow_val > 0 or precip_val > 10:
        return random.choices(["Medium", "Long"], weights=[0.5, 0.5])[0]
    if month in [12, 1, 2]:
        return random.choices(["Medium", "Long"], weights=[0.6, 0.4])[0]
    if is_weekend:
        return random.choices(["Short", "Medium"], weights=[0.7, 0.3])[0]
    if is_peak == 1:
        return random.choice(["Short", "Medium", "Long"])
    return None

# === Routes ===
@app.route('/', methods=['GET'])
def form():
    return render_template("index.html", stations=unique_stations)

@app.route('/predict', methods=['POST'])
def predict():
    date_str = request.form.get("date")
    time_str = request.form.get("time")
    station_name = request.form.get("station")
    input_time = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    hours = list(range(input_time.hour, 24))

    # Create records
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
            "Time_Category": "Morning" if hour < 12 else "Afternoon" if hour < 17 else "Evening",
            "Code": "UNKNOWN", "Bound": "N", "Line": "YU", "Vehicle": 0, "Min Gap": 0
        }
        records.append(row)

    df_input = pd.DataFrame(records)

    # Encode categorical features
    for col in ['Day', 'Station', 'Code', 'Bound', 'Line', 'Time_Category']:
        le = label_encoders[col]
        df_input[col] = df_input[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        df_input[col] = le.transform(df_input[col])

    # Reorder columns for model
    expected_order = ['Day', 'Station', 'Code', 'Min Gap', 'Bound', 'Line', 'Vehicle',
                      'Hour', 'Time_Category', 'Is_Weekend', 'Month', 'Is_Peak_Hour']
    features = df_input[expected_order]
    model_preds = model.predict(features)
    decoded_preds = target_encoder.inverse_transform(model_preds)

    # Apply rule overrides
    weather = fetch_weather(date_str)
    final_preds = []
    for i, row in df_input.iterrows():
        rule_based = apply_custom_rules(records[i], weather)
        final_preds.append(rule_based if rule_based else decoded_preds[i])

    # Prepare chart data
    chart_data = [{"hour": int(h), "delay": d} for h, d in zip(df_input["Hour"], final_preds)]

    # === Append to CSV for Power BI ===
    csv_file = "delay_predictions.csv"
    now_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_df = pd.DataFrame({
        "timestamp": [now_timestamp] * len(chart_data),
        "date": [date_str] * len(chart_data),
        "time": [f"{h['hour']}:00" for h in chart_data],
        "station": [station_name] * len(chart_data),
        "delay_category": [h["delay"] for h in chart_data]
    })
    if not os.path.isfile(csv_file):
        log_df.to_csv(csv_file, index=False)
    else:
        log_df.to_csv(csv_file, mode='a', header=False, index=False)

    # Get coordinates for map
    station_coords = station_coord_dict.get(station_name, [43.6532, -79.3832])

    return render_template(
        "index.html",
        stations=unique_stations,
        predictions=chart_data,
        chart_data=chart_data,
        station_coords=station_coords,
        date=date_str,
        station=station_name,
        weather=weather
    )

# === Run App ===
if __name__ == '__main__':
    app.run(debug=True)
