import os
import tempfile
import zipfile
import binascii
from reedsolo import RSCodec

# Configuration
DATA_CHUNK_SIZE = 205        # 223 (RS Data limit) - 16 (Nonce) - 2 (CRC) = 205 bytes
XOR_KEY = b"SecretKey123!"   # Key used for XOR encryption
SOURCE_DIR = "M:\supersecretfolder\secretsubfolder"

# Initialize Reed-Solomon Codec for 32 parity bytes (255 - 223 = 32)
rs = RSCodec(32)

def xor_encrypt_decrypt(data: bytes, key: bytes) -> bytes:
    """Encrypts or decrypts bytes using a repeating XOR key."""
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))

def calculate_crc16(data: bytes) -> bytes:
    """Computes standard CRC-16 (XMODEM/HQX) and returns it as 2 bytes."""
    crc = binascii.crc_hqx(data, 0)
    return crc.to_bytes(2, byteorder='big')

# 1. Search for the target .txt file
target_file_path = None
for root, dirs, files in os.walk(SOURCE_DIR):
    for file in files:
        if file.endswith(".txt") and not file.endswith("_framed.txt"):
            target_file_path = os.path.join(root, file)
            break
    if target_file_path:
        break

if target_file_path:
    framed_file_path = target_file_path.replace(".txt", "_framed.txt")
    frame_count = 0
    
    # 2. Process file layout
    with open(target_file_path, 'rb') as f_in, open(framed_file_path, 'wb') as f_out:
        while True:
            raw_payload = f_in.read(DATA_CHUNK_SIZE)
            if not raw_payload:
                break  # Reached end of file
            
            # If the final file chunk is smaller than 205 bytes, pad it with null bytes 
            # to keep a strict uniform 223-byte RS block dimension
            if len(raw_payload) < DATA_CHUNK_SIZE:
                raw_payload = raw_payload.ljust(DATA_CHUNK_SIZE, b'\x00')
            
            # A. Generate 16-byte random Nonce
            nonce_bytes = os.urandom(16)
            
            # B. Encrypt file data payload
            encrypted_payload = xor_encrypt_decrypt(raw_payload, XOR_KEY)
            
            # C. Calculate CRC-16 over Nonce + Encrypted Payload
            inner_content = nonce_bytes + encrypted_payload
            crc_bytes = calculate_crc16(inner_content)
            
            # Assemble the exact 223-byte baseline block
            rs_data_block = inner_content + crc_bytes  # 16 + 205 + 2 = 223 bytes
            
            # D. Apply Reed-Solomon FEC (Appends 32 parity bytes to the end)
            final_rs_frame = rs.encode(rs_data_block)  # Total: 255 bytes
            
            # Write out the completed 255-byte transmission frame
            f_out.write(final_rs_frame)
            frame_count += 1
            
    print(f"Data split into {frame_count} frames protected by Reed-Solomon (255,223).")

    # 3. Compress the finalized framed file into a .zip archive
    zip_file_path = target_file_path.replace(".txt", ".zip")
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(framed_file_path, arcname=os.path.basename(framed_file_path))
    
    print(f"Compressed framed/encrypted file to: {zip_file_path}")
    os.remove(framed_file_path)

    # 4. Pass the ZIP path to the temporary file for Script 2
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt')
    temp_file.write(zip_file_path)
    temp_file.close()

    print(f"Temp file created at: {temp_file.name}")
else:
    print("No target .txt file found.")
