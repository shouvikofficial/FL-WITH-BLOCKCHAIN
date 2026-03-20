from web3 import Web3
import os

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
if w3.is_connected():
    contract = w3.eth.contract(address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=ABI)
    try:
        events = contract.events.UpdateLogged.create_filter(from_block=0).get_all_entries()
        for e in events[-1:]:
            print(e.args)
            print(e.blockHash.hex())
            print(e.transactionHash.hex())
        print("Success!")
    except Exception as e:
        print("Error:", e)
