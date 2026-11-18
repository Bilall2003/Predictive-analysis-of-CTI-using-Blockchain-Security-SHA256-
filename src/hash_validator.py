
import re

# A hash is just hexadecimal text of a fixed length: 32 chars = MD5, 40 = SHA-1, 64 = SHA-256
HASH_PATTERNS = {
    'MD5': re.compile(r'^[a-fA-F0-9]{32}$'),
    'SHA-1': re.compile(r'^[a-fA-F0-9]{40}$'),
    'SHA-256': re.compile(r'^[a-fA-F0-9]{64}$')
}


def validate_hash_format(hash_str: str) -> dict:
    """Return {"valid": bool, "type": 'MD5' | 'SHA-1' | 'SHA-256' | 'Invalid Format' | 'Unknown'}."""
    if not hash_str or not isinstance(hash_str, str):
        return {"valid": False, "type": "Unknown"}

    hash_str = hash_str.strip()
    for algorithm, pattern in HASH_PATTERNS.items():
        if pattern.match(hash_str):
            return {"valid": True, "type": algorithm}

    return {"valid": False, "type": "Invalid Format"}


def is_blacklisted_hash(hash_str: str) -> bool:
   
    if not validate_hash_format(hash_str)["valid"]:
        return False

    return False  # placeholder: no threat-intelligence database connected
