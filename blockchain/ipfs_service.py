import hashlib
import time
import pickle

# IPFS Base58 alphabet
ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

def base58_encode(num):
    """Encode a number in Base58."""
    encode = ''
    if num < 0:
        return ''
    while num >= 58:
        num, mod = divmod(num, 58)
        encode = ALPHABET[mod] + encode
    if num:
        encode = ALPHABET[num] + encode
    return encode

def simulated_ipfs_pin(data_dict, filename="global_model.pkl"):
    """
    Simulates pinning a file to IPFS.
    It takes the data (weights), creates a SHA-256 hash, and wraps it
    in the IPFS Multihash format (0x12, 0x20...) then Base58 encodes it.
    This creates an authentic-looking CID starting with 'Qm...'.
    """
    # 1. Serialize data (Using pickle instead of JSON to handle NumPy arrays perfectly)
    raw_bytes = pickle.dumps(data_dict)
    
    # 2. SHA-256 Hash
    sha256_hash = hashlib.sha256(raw_bytes).digest()
    
    # 3. Prepend IPFS Multihash metadata 
    # 0x12 = SHA256, 0x20 = length 32 bytes
    ipfs_multihash = b'\x12\x20' + sha256_hash
    
    # 4. Convert to integer for Base58 encoding
    hash_int = int.from_bytes(ipfs_multihash, byteorder='big')
    
    # 5. Encode to Base58
    cid = base58_encode(hash_int)
    
    print(f"\n  🖧 [IPFS] Uploading '{filename}' to Decentralized Storage...")
    time.sleep(1.5) # Simulate network upload delay
    
    # --- PHYSICAL STORAGE SIMULATION ---
    # We save the actual file to a local "IPFS Node" folder, named by its cryptographic CID!
    import os
    ipfs_dir = os.path.join(os.getcwd(), "ipfs_node_storage")
    os.makedirs(ipfs_dir, exist_ok=True)
    filepath = os.path.join(ipfs_dir, f"{cid}.pkl")
    
    with open(filepath, "wb") as f:
        f.write(raw_bytes)
    # -----------------------------------
    
    print(f"  ✅ [IPFS] Pinning Complete! CID: {cid}")
    print(f"  📂 [IPFS] You can view your saved model in the 'ipfs_node_storage' folder!")
    
    return cid
