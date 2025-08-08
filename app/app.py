from flask import Flask, render_template, jsonify
from config import Config
import json

app = Flask(__name__)
app.config.from_object(Config)

@app.route("/")
def dashboard():
    with open(Config.RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    suspicious = [entry for entry in data if entry.get("label", {}).get("label") == "drug-related"]
    return render_template("dashboard.html", data=data, suspicious=suspicious)

@app.route("/api/data")
def get_data():
    with open(Config.RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)