from flask import Flask
from web3 import Web3

app = Flask(__name__)
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))

@app.route("/")
def home():
    return """
    <h2>Blockchain-Federated Learning Dashboard</h2>
    <p>Status: Running</p>
    <p>Blockchain Connected</p>
    """

app.run(host="0.0.0.0", port=5000)
