import time
import os
import sys
import json
import numpy as np
import requests
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Reuse the core logic from your existing client
from real_client import run

if __name__ == "__main__":
    print("======================================================")
    print("   🏥 Federated Learning Client Startup 🏥")
    print("======================================================\n")

    client_id = input("1. Enter your Client ID (e.g., Client_1, Client_2): ").strip()
    if not client_id.startswith("Client_"):
        print("❌ Error: Client ID must start with 'Client_'")
        input("Press Enter to exit...")
        sys.exit(1)

    server_url = input("2. Enter the Server URL (e.g., https://abc.ngrok-free.app): ").strip().rstrip("/")
    if not server_url.startswith("http"):
        print("❌ Error: Server URL must start with http:// or https://")
        input("Press Enter to exit...")
        sys.exit(1)

    data_path = input("3. Enter the path to your Excel dataset (e.g., data/dataset.xlsx): ").strip()
    if not os.path.exists(data_path):
        print(f"❌ Error: Cannot find file at '{data_path}'")
        input("Press Enter to exit...")
        sys.exit(1)

    print("\n✅ All inputs accepted. Connecting to Server...\n")
    
    # Start the actual runs
    try:
        run(client_id=client_id, 
            server_url=server_url, 
            data_path=data_path, 
            label_col="target", 
            total_clients=3, 
            total_rounds=25)
            
        print("\n🎉 Training session complete. You can now close this window.")
        input("Press Enter to exit...")
    except Exception as e:
        print(f"\n❌ A fatal error occurred: {e}")
        input("Press Enter to exit...")
