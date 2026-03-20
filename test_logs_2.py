from web3 import Web3

GANACHE_URL = "http://127.0.0.1:7545"
CONTRACT_ADDRESS = "0x145699CAb33EC0f7be2AC1a192AA6785a1B6cbb1"

ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "string", "name": "clientId", "type": "string"},
            {"indexed": False, "internalType": "string", "name": "modelHash", "type": "string"},
            {"indexed": False, "internalType": "uint256", "name": "roundId", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"}
        ],
        "name": "UpdateLogged",
        "type": "event"
    }
]

w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
contract = w3.eth.contract(address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=ABI)

try:
    raw_logs = w3.eth.get_logs({'fromBlock': 0, 'address': contract.address})
    print(f"Got {len(raw_logs)} raw logs.")
    
    events = []
    for log in raw_logs:
        try:
            parsed = contract.events.UpdateLogged().process_log(log)
            events.append(parsed)
        except Exception:
            pass
            
    print(f"Parsed {len(events)} events.")
    if events:
        print(events[0].blockHash.hex())
except Exception as e:
    print("Error:", e)
