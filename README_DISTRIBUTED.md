# 🌐 Real Distributed Federated Learning — Setup Guide

This guide explains how to connect 3 real PCs for distributed federated learning.

---

## 🖥️ SERVER SETUP (your main PC)

1. Start the Flask server:
```bash
venv\Scripts\python.exe app.py
```

2. In a separate terminal, start ngrok:
```bash
ngrok http 5000
```

3. Copy the ngrok URL (e.g. `https://abc123.ngrok-free.app`)

4. Click **"Start Federation"** on the dashboard — this initializes the FL session and waits for 3 clients.

---

## 💻 CLIENT SETUP (each of the 3 PCs)

### Step 1: Copy these files to each client PC
```
real_client.py
client_requirements.txt
data/dataset.xlsx       ← each client should have their own data shard ideally
```

### Step 2: Install dependencies
```bash
pip install -r client_requirements.txt
```

### Step 3: Run the client script
Replace `YOUR_NGROK_URL` with the URL from ngrok.

**PC 1:**
```bash
python real_client.py --client_id Client_1 --server https://YOUR_NGROK_URL
```

**PC 2:**
```bash
python real_client.py --client_id Client_2 --server https://YOUR_NGROK_URL
```

**PC 3:**
```bash
python real_client.py --client_id Client_3 --server https://YOUR_NGROK_URL
```

---

## 🔄 How It Works

```
Each round:
1. Client downloads global model from server          (GET /fl/get_global_model)
2. Client trains locally on its own data
3. Client sends weights to server                     (POST /fl/submit_update)
4. Server waits for ALL 3 clients → then aggregates
5. Server logs to blockchain, updates dashboard
6. Repeat for 25 rounds
```

---

## 📊 Monitor Training

Watch the dashboard live at: `https://YOUR_NGROK_URL`

- **System Status** shows how many clients have submitted each round
- **Blockchain Logs** tab shows on-chain records
- **Attack Audit** tab shows any detected malicious clients

---

## ⚙️ Configuration

You can customize via environment variables:

| Variable          | Default | Description                  |
|-------------------|---------|------------------------------|
| `FL_NUM_CLIENTS`  | 3       | Number of expected clients   |
| `FL_ROUNDS`       | 25      | Total training rounds        |
| `FL_ROUND_TIMEOUT`| 300     | Seconds to wait per round    |

Example:
```bash
FL_NUM_CLIENTS=5 FL_ROUNDS=10 venv\Scripts\python.exe app.py
```
