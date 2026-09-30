import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
model = joblib.load(os.path.join("model", "birth_model.joblib"))

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head><title>India Birth Rate Predictor</title></head>
<body style="font-family: sans-serif; max-width: 500px; margin: 50px auto;">
  <h2>India Birth Rate Predictor</h2>
  <p>Predicts crude birth rate (births per 1000 people).</p>
  <form action="/predict" method="get">
    <label>Enter Year: </label>
    <input type="number" name="year" value="{{ year or 2025 }}" required>
    <button type="submit">Predict</button>
  </form>
  {% if prediction is not none %}
    <h3>Predicted birth rate in {{ year }}: {{ prediction }} per 1000 people</h3>
  {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_PAGE, prediction=None, year=None)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    data = request.get_json(silent=True) or request.args
    year = data.get("year")
    if year is None:
        return jsonify({"error": "Provide 'year'"}), 400
    year = int(year)
    pred = round(float(model.predict(np.array([[year]]))[0]), 2)
    if request.method == "GET" and request.accept_mimetypes.accept_html:
        return render_template_string(HTML_PAGE, prediction=pred, year=year)
    return jsonify({"year": year, "predicted_birth_rate": pred})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
