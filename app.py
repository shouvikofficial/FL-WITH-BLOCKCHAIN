import os
import json
import subprocess
from flask import Flask, render_template, jsonify, send_from_directory

app = Flask(__name__)

# Register the real distributed FL API routes
try:
    from fl_server_api import fl_api
    app.register_blueprint(fl_api)
    print("✅ Real distributed FL API registered at /fl/...")
except Exception as e:
    print(f"⚠️  FL API not loaded: {e}")

# Ensure static directory exists for the metrics file
os.makedirs("static", exist_ok=True)
METRICS_FILE = "static/metrics.json"

# Global reference to the training process
training_process = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/blockchain_log")
def blockchain_log():
    return render_template("blockchain_log.html")

@app.route("/attack_log")
def attack_log():
    return render_template("attack_log.html")

@app.route("/api/start", methods=["POST"])
def start_training():
    """
    Initializes the real distributed FL session so real clients can connect.
    Does NOT run the local simulation (run.py).
    """
    try:
        from fl_server_api import reset_fl_session
        reset_fl_session()
        return jsonify({"success": True, "message": "Real FL session started. Waiting for clients..."})
    except Exception as e:
        print(f"⚠️  Could not reset FL state: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/stop", methods=["POST"])
def stop_training_api():
    """
    Kills the background Federated Learning script.
    """
    global training_process
    if training_process and training_process.poll() is None:
        training_process.terminate()
        try:
            training_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            training_process.kill()
        
        training_process = None
        
        # Mark as stopped in metrics
        try:
            with open(METRICS_FILE, "r") as f:
                data = json.load(f)
            data["status"] = "Training Stopped Manually 🛑"
            data["completed"] = True
            with open(METRICS_FILE, "w") as f:
                json.dump(data, f)
        except Exception:
            pass
            
        return jsonify({"success": True, "message": "Training stopped."})
        
    return jsonify({"success": False, "message": "No training in progress."})

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

@app.route("/api/blockchain_logs", methods=["GET"])
def get_blockchain_logs():
    """
    Fetches all logs from the smart contract via blockchain.py.
    """
    try:
        from blockchain.blockchain import get_all_updates
        logs = get_all_updates()
        # logs is a list of dicts: [{'client': ..., 'hash': ..., 'timestamp': ..., 'round': ...}, ...]
        return jsonify({"success": True, "logs": logs})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Web Dashboard on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
