import os
import json
import subprocess
from flask import Flask, render_template, jsonify, send_from_directory, request

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

@app.route("/client_dashboard")
def client_dashboard():
    return render_template("client_dashboard.html")

@app.route("/api/client_event", methods=["POST"])
def receive_client_event():
    """
    REAL DISTRIBUTED USE: called by real_client.py running on a remote PC.
    Each event is POSTed here and appended server-side to the client's own log file.
    This means the dashboard can show live data from clients on ANY machine.
    
    Body: { "client_id": "Client_1", "event": { "type": "...", "timestamp": ..., ... } }
    Or for session reset: { "client_id": "Client_1", "reset": true }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    client_id = data.get("client_id", "").strip()
    if not client_id:
        return jsonify({"error": "Missing client_id"}), 400

    # Sanitise to prevent path traversal
    safe_id  = client_id.replace("/", "_").replace("\\", "_").replace("..", "_")
    log_path = f"static/client_training_log_{safe_id}.json"
    os.makedirs("static", exist_ok=True)

    # Session reset — wipe old log for fresh run
    if data.get("reset"):
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        return jsonify({"status": "reset", "client_id": client_id})

    # Append event
    event = data.get("event")
    if not event:
        return jsonify({"error": "Missing event"}), 400

    # Read existing events
    events = []
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                events = json.load(f)
        except (json.JSONDecodeError, IOError):
            events = []

    events.append(event)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(events, f)

    return jsonify({"status": "ok", "client_id": client_id, "total_events": len(events)})


@app.route("/api/client_log", methods=["GET"])
def get_client_log():
    """
    Reads the per-client training log emitted by real_client.py.
    Requires ?client_id=Client_1  (or whichever client ID)
    Each client writes its own isolated file: client_training_log_Client_1.json
    """
    client_id = request.args.get("client_id", "").strip()
    if not client_id:
        return jsonify({"events": [], "status": "missing_client_id",
                        "error": "Pass ?client_id=Client_1 in the URL"}), 400

    # Sanitise to prevent path traversal
    safe_id   = client_id.replace("/", "_").replace("\\", "_").replace("..", "_")
    log_path  = f"static/client_training_log_{safe_id}.json"

    if not os.path.exists(log_path):
        return jsonify({"events": [], "status": "no_session",
                        "client_id": client_id})
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            events = json.load(f)
        return jsonify({"events": events, "status": "ok", "client_id": client_id})
    except (json.JSONDecodeError, IOError):
        return jsonify({"events": [], "status": "reading", "client_id": client_id})

@app.route("/api/client_sessions", methods=["GET"])
def list_client_sessions():
    """Return a list of client IDs that have active log files in static/."""
    import glob
    files = glob.glob("static/client_training_log_*.json")
    client_ids = []
    for f in files:
        name = os.path.basename(f)       # client_training_log_Client_1.json
        cid  = name[len("client_training_log_"):-len(".json")]
        client_ids.append(cid)
    return jsonify({"clients": sorted(client_ids)})


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
