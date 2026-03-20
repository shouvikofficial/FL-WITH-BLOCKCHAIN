from web3 import Web3
import hashlib
import numpy as np
import os
import time
from typing import List, Dict, Optional

# ======================================================
# CONFIGURATION
# ======================================================
GANACHE_URL = os.getenv("BLOCKCHAIN_RPC", "http://127.0.0.1:7545")
CONTRACT_ADDRESS = os.getenv(
    "CONTRACT_ADDRESS",
    "0x145699CAb33EC0f7be2AC1a192AA6785a1B6cbb1"
)

EXPECTED_CHAIN_ID = int(os.getenv("CHAIN_ID", 1337))
DEFAULT_GAS_LIMIT = int(os.getenv("GAS_LIMIT", 3_000_000))
ACCOUNT_INDEX = int(os.getenv("ACCOUNT_INDEX", 0))

MAX_RETRIES = 3
RETRY_DELAY = 1.5

# ======================================================
# CONNECT TO BLOCKCHAIN
# ======================================================
w3 = Web3(Web3.HTTPProvider(GANACHE_URL))

if not w3.is_connected():
    raise RuntimeError("❌ Blockchain RPC not reachable")

if w3.eth.chain_id != EXPECTED_CHAIN_ID:
    raise RuntimeError(
        f"❌ Wrong chain ID {w3.eth.chain_id}, expected {EXPECTED_CHAIN_ID}"
    )

ACCOUNT = w3.eth.accounts[ACCOUNT_INDEX]
print(f"✅ Blockchain connected | Chain ID={EXPECTED_CHAIN_ID} | Account={ACCOUNT}")

# ======================================================
# CONTRACT
# ======================================================
import json as _json
import pathlib as _pathlib

# Load ABI from file (keeps blockchain.py clean)
_ABI_PATH = _pathlib.Path(__file__).parent / "UpdateLogABI.json"
if _ABI_PATH.exists():
    with open(_ABI_PATH) as _f:
        ABI = _json.load(_f)
else:
    # Fallback minimal ABI
    ABI = [
        {
            "inputs": [
                {"internalType": "string", "name": "cid", "type": "string"},
                {"internalType": "string", "name": "h",   "type": "string"}
            ],
            "name": "logUpdate",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [],
            "name": "totalUpdates",
            "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function"
        },
    ]

contract = w3.eth.contract(
    address=Web3.to_checksum_address(CONTRACT_ADDRESS),
    abi=ABI
)

# ======================================================
# HASHING (PRIVACY-PRESERVING)
# ======================================================
def hash_weights(weights: List[np.ndarray], client_id: str, round_id: int) -> str:
    if round_id < 1:
        raise ValueError("round_id must be >= 1")

    if not client_id or not isinstance(client_id, str):
        raise ValueError("client_id must be a non-empty string")

    flat = np.concatenate([w.flatten() for w in weights])
    context = f"{client_id}|round={round_id}".encode()

    payload = np.concatenate([
        flat,
        np.frombuffer(context, dtype=np.uint8)
    ])

    return hashlib.sha256(payload.tobytes()).hexdigest()

# ======================================================
# TRANSACTION SENDER
# ======================================================
def _send_tx(func_call) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            tx_hash = func_call.transact({
                "from": ACCOUNT,
                "gas": DEFAULT_GAS_LIMIT
            })
            return tx_hash.hex()
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(e)
            time.sleep(RETRY_DELAY)

# ======================================================
# PUBLIC LOG FUNCTION
# ======================================================
def log_update(
    client_id: str,
    weights: List[np.ndarray],
    round_id: int,
    wait_for_receipt: bool = True,
    dry_run: bool = False
) -> Optional[Dict]:
    """
    Log a federated learning update to blockchain.

    dry_run=True → hash only, no blockchain write (testing / attacks)
    """

    model_hash = hash_weights(weights, client_id, round_id)

    if dry_run:
        print(f"🧪 DRY RUN | Client={client_id} | Round={round_id}")
        return {
            "client": client_id,
            "round": round_id,
            "hash": model_hash,
            "tx_hash": None,
            "block": None
        }

    try:
        tx_hash = _send_tx(
            contract.functions.logUpdate(client_id, model_hash)
        )

        block_number = None
        if wait_for_receipt:
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
            block_number = receipt.blockNumber
            print(
                f"🔗 LOGGED | Client={client_id} | "
                f"Round={round_id} | Block={block_number}"
            )

        return {
            "client": client_id,
            "round": round_id,
            "hash": model_hash,
            "tx_hash": tx_hash,
            "block": block_number
        }

    except Exception as e:
        print(f"❌ BLOCKCHAIN ERROR | Client={client_id} | {e}")
        return None

# ======================================================
# READ-ONLY AUDIT
# ======================================================
def get_total_updates() -> int:
    """Total number of updates stored on-chain."""
    try:
        if contract is None:
            return 0
        return contract.functions.totalUpdates().call()
    except Exception as e:
        print(f"❌ Error getting total updates: {e}")
        return 0

_updates_cache = []

def get_all_updates() -> List[Dict]:
    """Fetch all stored updates from the blockchain using a cache."""
    global _updates_cache
    try:
        total = get_total_updates()
        if total == len(_updates_cache):
            return list(_updates_cache)
            
        raw_logs = w3.eth.get_logs({'fromBlock': 0, 'address': contract.address})
        events = []
        for log in raw_logs:
            try:
                events.append(contract.events.UpdateLogged().process_log(log))
            except Exception:
                pass
                
        start_idx = len(_updates_cache)
        for i in range(start_idx, total):
            update = contract.functions.getUpdate(i).call()
            e = events[i] if i < len(events) else None
            
            block_hash = e.blockHash.hex() if e and e.blockHash else "Unknown"
            tx_hash = e.transactionHash.hex() if e and e.transactionHash else "Unknown"
            
            _updates_cache.append({
                "client": update[0],
                "hash": update[1],
                "timestamp": update[2],
                "round": update[3],
                "block_hash": block_hash,
                "tx_hash": tx_hash
            })
            
        return list(_updates_cache)
    except Exception as e:
        print(f"❌ Error fetching updates: {e}")
        return list(_updates_cache)
