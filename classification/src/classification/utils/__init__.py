import struct

import numpy as np

from common.defaults import MELVEC_LENGTH, N_MELVECS

'''
#16 bits
def payload_to_melvecs(
    payload: str, melvec_length: int = MELVEC_LENGTH, n_melvecs: int = N_MELVECS
) -> np.ndarray:
    """Convert a payload string to a melvecs array."""
    fmt = f"!{melvec_length}h"
    buffer = bytes.fromhex(payload.strip())
    unpacked = struct.iter_unpack(fmt, buffer)
    melvecs_q15int = np.asarray(list(unpacked), dtype=np.int16)
    melvecs = melvecs_q15int.astype(float) / 32768  # 32768 = 2 ** 15
    melvecs = np.rot90(melvecs, k=-1, axes=(0, 1))
    melvecs = np.fliplr(melvecs)
    return melvecs
'''

def payload_to_melvecs(
    payload: str, melvec_length: int = MELVEC_LENGTH, n_melvecs: int = N_MELVECS
) -> np.ndarray:
    """Convert a payload of 8-bit values to a 1D array."""
    buffer = bytes.fromhex(payload.strip())

    expected_values = melvec_length * n_melvecs  # 100 if 10x10
    expected_bytes = expected_values  # 1 byte per value
    if len(buffer) != expected_bytes:
        raise ValueError(f"Expected {expected_bytes} bytes, got {len(buffer)}")

    melvecs = np.frombuffer(buffer, dtype=np.uint8)  # Or np.int8 if signed
    return melvecs




'''
def payload_to_melvecs(
    payload: str, melvec_length: int = MELVEC_LENGTH, n_melvecs: int = N_MELVECS
) -> np.ndarray:
    """Convert a payload string to a melvecs array (8-bit version)."""
    fmt = f"!{melvec_length}B" #Déterminer si unsigned (B) ou signed (b)
    buffer = bytes.fromhex(payload.strip())
    unpacked = struct.iter_unpack(fmt, buffer)
    melvecs = np.asarray(list(unpacked), dtype=np.uint8)
    melvecs = np.rot90(melvecs, k=-1, axes=(0, 1))
    melvecs = np.fliplr(melvecs)
    return melvecs
'''
