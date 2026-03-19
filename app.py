import os
import json
import subprocess
from flask import Flask, render_template, jsonify, send_from_directory

app = Flask(__name__)

# Ensure static directory exists for the metrics file
os.makedirs("static", exist_ok=True)
METRICS_FILE = "static/metrics.json"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/start", methods=["POST"])
def start_training():
    """
    Triggers the Federated Learning script in the background.
    Resets the metrics JSON first.
    """
    # Reset metrics file
    initial_data = {
        "status": "Starting Initialization...",
        "round": 0,
        "total_rounds": 25, # Pull from config ideally, hardcoded 25 for demo
        "metrics": [],
        "malicious_attackers": [],
        "completed": False
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(initial_data, f)

    # Launch in background so we don't block the request
    # Use python executable from the virtual environment
    python_exe = os.path.join("venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = "python" # fallback

    try:
        subprocess.Popen([python_exe, "run.py"])
        return jsonify({"success": True, "message": "Federated Learning started natively."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """
    Reads the latest metrics dumped by server.py and serves it to the frontend.
    """
    if not os.path.exists(METRICS_FILE):
        return jsonify({"status": "Waiting for simulation to start... (No metrics yet)", "round": 0})
        
    try:
        with open(METRICS_FILE, "r") as f:
            data = json.load(f)
        return jsonify(data)
    except json.JSONDecodeError:
        # File might be mid-write
        return jsonify({"status": "Parsing metrics...", "round": -1})

if __name__ == "__main__":
    print("🚀 Starting Web Dashboard on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
