"""
California Housing Price Predictor - Interactive Web Application
Built with Flask, Scikit-Learn, and Joblib.
"""

import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from sklearn.base import BaseEstimator, TransformerMixin

# ==============================================================================
# 1. Custom Transformer Definition (required by the saved scikit-learn pipeline)
# ==============================================================================
import sys
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# Inject into __main__ so joblib/pickle unpickler always finds it
setattr(sys.modules['__main__'], 'CombinedAttributesAdder', CombinedAttributesAdder)


# ==============================================================================
# 2. Load Model Pipeline
# ==============================================================================
def find_model_path():
    candidates = [
        "my_california_housing_model.pkl",
        os.path.join("..", "my_california_housing_model.pkl"),
        os.path.join(os.path.dirname(__file__), "my_california_housing_model.pkl")
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return "my_california_housing_model.pkl"

model_path = find_model_path()
print(f"Loading model from: {model_path} ...")
model = joblib.load(model_path)
print("Model loaded successfully!")

# ==============================================================================
# 3. Flask Web Application & Interactive UI
# ==============================================================================
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>California Housing Price Predictor</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --accent: #10b981;
      --input-bg: #0f172a;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 24px;
    }
    .container {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      max-width: 860px;
      width: 100%;
      padding: 32px;
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    }
    .header {
      margin-bottom: 24px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
    }
    .header h1 {
      font-size: 1.6rem;
      font-weight: 700;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .header p {
      color: var(--text-muted);
      font-size: 0.92rem;
      margin-top: 4px;
    }
    .presets {
      display: flex;
      gap: 10px;
      margin-bottom: 24px;
      flex-wrap: wrap;
      align-items: center;
    }
    .presets span {
      font-size: 0.85rem;
      color: var(--text-muted);
      font-weight: 600;
    }
    .preset-btn {
      background: #334155;
      color: var(--text);
      border: 1px solid #475569;
      padding: 6px 14px;
      border-radius: 8px;
      font-size: 0.82rem;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .preset-btn:hover {
      background: var(--primary);
      border-color: var(--primary);
    }
    .form-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 18px;
    }
    .field-group {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .field-group label {
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .field-group input, .field-group select {
      background: var(--input-bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 10px 14px;
      border-radius: 8px;
      font-size: 0.95rem;
      transition: border-color 0.2s ease;
      outline: none;
    }
    .field-group input:focus, .field-group select:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25);
    }
    .actions {
      margin-top: 28px;
      display: flex;
      gap: 16px;
      align-items: center;
    }
    .submit-btn {
      background: var(--primary);
      color: #fff;
      font-weight: 600;
      padding: 14px 28px;
      border-radius: 10px;
      border: none;
      font-size: 1rem;
      cursor: pointer;
      transition: background 0.2s ease;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .submit-btn:hover {
      background: var(--primary-hover);
    }
    .result-card {
      margin-top: 24px;
      padding: 20px 24px;
      border-radius: 12px;
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(16, 185, 129, 0.3);
      display: none;
      animation: fadeIn 0.3s ease;
    }
    .result-card.active {
      display: block;
    }
    .result-title {
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--accent);
      font-weight: 700;
      margin-bottom: 4px;
    }
    .result-value {
      font-size: 2.2rem;
      font-weight: 800;
      color: #fff;
    }
    .result-sub {
      color: var(--text-muted);
      font-size: 0.85rem;
      margin-top: 4px;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🏠 California Housing Price Predictor</h1>
      <p>Enter housing district features below to get a real-time price estimation from the trained Scikit-Learn model.</p>
    </div>

    <div class="presets">
      <span>Quick Presets:</span>
      <button type="button" class="preset-btn" onclick="applyPreset('bay_area')">San Francisco Bay Area</button>
      <button type="button" class="preset-btn" onclick="applyPreset('inland')">Inland Valley</button>
      <button type="button" class="preset-btn" onclick="applyPreset('coastal')">Coastal District</button>
    </div>

    <form id="predictionForm" onsubmit="handlePredict(event)">
      <div class="form-grid">
        <div class="field-group">
          <label for="longitude">Longitude</label>
          <input type="number" step="0.01" id="longitude" name="longitude" value="-122.23" required>
        </div>
        <div class="field-group">
          <label for="latitude">Latitude</label>
          <input type="number" step="0.01" id="latitude" name="latitude" value="37.88" required>
        </div>
        <div class="field-group">
          <label for="housing_median_age">Median House Age</label>
          <input type="number" step="1" id="housing_median_age" name="housing_median_age" value="41" required>
        </div>
        <div class="field-group">
          <label for="total_rooms">Total Rooms</label>
          <input type="number" step="1" id="total_rooms" name="total_rooms" value="880" required>
        </div>
        <div class="field-group">
          <label for="total_bedrooms">Total Bedrooms</label>
          <input type="number" step="1" id="total_bedrooms" name="total_bedrooms" value="129" required>
        </div>
        <div class="field-group">
          <label for="population">District Population</label>
          <input type="number" step="1" id="population" name="population" value="322" required>
        </div>
        <div class="field-group">
          <label for="households">Total Households</label>
          <input type="number" step="1" id="households" name="households" value="126" required>
        </div>
        <div class="field-group">
          <label for="median_income">Median Income ($10k units)</label>
          <input type="number" step="0.0001" id="median_income" name="median_income" value="8.3252" required title="e.g. 8.32 = $83,200/year">
        </div>
        <div class="field-group">
          <label for="ocean_proximity">Ocean Proximity</label>
          <select id="ocean_proximity" name="ocean_proximity" required>
            <option value="NEAR BAY" selected>NEAR BAY</option>
            <option value="<1H OCEAN">&lt;1H OCEAN</option>
            <option value="INLAND">INLAND</option>
            <option value="NEAR OCEAN">NEAR OCEAN</option>
            <option value="ISLAND">ISLAND</option>
          </select>
        </div>
      </div>

      <div class="actions">
        <button type="submit" class="submit-btn" id="btnSubmit">
          <span>⚡ Calculate Predicted Price</span>
        </button>
      </div>
    </form>

    <div class="result-card" id="resultCard">
      <div class="result-title">Estimated Median House Value</div>
      <div class="result-value" id="resultPrice">$0</div>
      <div class="result-sub" id="resultNote">Predicted using Scikit-Learn RandomForest / Preparation Pipeline</div>
    </div>
  </div>

  <script>
    const presets = {
      bay_area: {
        longitude: -122.23, latitude: 37.88, housing_median_age: 41,
        total_rooms: 880, total_bedrooms: 129, population: 322,
        households: 126, median_income: 8.3252, ocean_proximity: 'NEAR BAY'
      },
      inland: {
        longitude: -119.80, latitude: 36.75, housing_median_age: 25,
        total_rooms: 2450, total_bedrooms: 480, population: 1400,
        households: 450, median_income: 2.8500, ocean_proximity: 'INLAND'
      },
      coastal: {
        longitude: -118.50, latitude: 34.02, housing_median_age: 32,
        total_rooms: 3100, total_bedrooms: 520, population: 1100,
        households: 490, median_income: 6.7500, ocean_proximity: '<1H OCEAN'
      }
    };

    function applyPreset(name) {
      const data = presets[name];
      if (!data) return;
      for (const [key, val] of Object.entries(data)) {
        const input = document.getElementById(key);
        if (input) input.value = val;
      }
      document.getElementById('resultCard').classList.remove('active');
    }

    async function handlePredict(e) {
      e.preventDefault();
      const btn = document.getElementById('btnSubmit');
      btn.innerText = 'Calculating...';
      btn.disabled = true;

      const payload = {
        longitude: parseFloat(document.getElementById('longitude').value),
        latitude: parseFloat(document.getElementById('latitude').value),
        housing_median_age: parseFloat(document.getElementById('housing_median_age').value),
        total_rooms: parseFloat(document.getElementById('total_rooms').value),
        total_bedrooms: parseFloat(document.getElementById('total_bedrooms').value),
        population: parseFloat(document.getElementById('population').value),
        households: parseFloat(document.getElementById('households').value),
        median_income: parseFloat(document.getElementById('median_income').value),
        ocean_proximity: document.getElementById('ocean_proximity').value
      };

      try {
        const response = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (data.status === 'success') {
          document.getElementById('resultPrice').innerText = '$' + data.predicted_price.toLocaleString('en-US', {
            minimumFractionDigits: 2, maximumFractionDigits: 2
          });
          document.getElementById('resultCard').classList.add('active');
        } else {
          alert('Error: ' + data.message);
        }
      } catch (err) {
        alert('Failed to connect to prediction server: ' + err);
      } finally {
        btn.innerText = '⚡ Calculate Predicted Price';
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        # Create single-row DataFrame with expected columns
        input_df = pd.DataFrame([{
            'longitude': float(data['longitude']),
            'latitude': float(data['latitude']),
            'housing_median_age': float(data['housing_median_age']),
            'total_rooms': float(data['total_rooms']),
            'total_bedrooms': float(data['total_bedrooms']),
            'population': float(data['population']),
            'households': float(data['households']),
            'median_income': float(data['median_income']),
            'ocean_proximity': str(data['ocean_proximity'])
        }])

        prediction = model.predict(input_df)[0]
        return jsonify({
            'status': 'success',
            'predicted_price': round(float(prediction), 2)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print(" California Housing Prediction Web App")
    print(f" Server running at: http://0.0.0.0:{port}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)
