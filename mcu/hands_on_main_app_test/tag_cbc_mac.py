import numpy as np
from AES import encrypt, decrypt, Mode

AES_key = bytes.fromhex("0123456789abcdef0123456789abcdef")


msg = "00000020000000007fff7fff7fff7fff7fff7fff7fff7fff7fff7fff7fff7fff7fff7fff7fff7fff"


def tag_cbc_mac(msg, AES_KEY):
    statew = np.zeros(4, dtype=np.uint32)

    n = len(msg) // 16
    if len(msg) % 16 != 0:
        n += 1
    for i in range(n):
        for j in range(16):
            if (i == n-1) and (j>=len(msg)%16):
                state[j] = 0
            else:
                state[j] = msg[i*16+j]

                ### Unfinished



print(tag_cbc_mac(msg, AES_key))