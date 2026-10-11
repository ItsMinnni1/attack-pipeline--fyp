import os
import zipfile
import binascii
import subprocess
from reedsolo import RSCodec, ReedSolomonError

FRAME_SIZE = 255             # Entire RS encoded block size
DATA_BLOCK_SIZE = 223        # RS original data size (255 - 32 parity)
XOR_KEY = b"SecretKey123!"   # Key used for XOR decryption
TEMP_TEXT_PATH = "recovered_output.txt"  # File to pass to Notepad

rs = RSCodec(32)

def xor_encrypt_decrypt(data: bytes, key: bytes) -> bytes:
    """Decrypts bytes using the repeating XOR key."""
    return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))

def calculate_crc16(data: bytes) -> bytes:
    """Computes standard CRC-16 (XMODEM/HQX) and returns it as 2 bytes."""
    crc = binascii.crc_hqx(data, 0)
    return crc.to_bytes(2, byteorder='big')

temp_pointer_file = r"C:\Users\POTATO~1\AppData\Local\Temp\tmpcauz2ht2.txt" 

if not os.path.exists(temp_pointer_file):
    print(f"Error: Temporary pointer file '{temp_pointer_file}' not found.")
    exit(1)

with open(temp_pointer_file, 'r') as f:
    zip_file_path = f.read().strip()

if not os.path.exists(zip_file_path):
    print(f"Error: Target zip file '{zip_file_path}' does not exist.")
    exit(1)

working_dir = os.path.dirname(zip_file_path)
with zipfile.ZipFile(zip_file_path, 'r') as zipf:
    internal_filename = zipf.namelist()[0]
    zipf.extract(internal_filename, path=working_dir)

extracted_framed_file = os.path.join(working_dir, internal_filename)

# 3. Process the file frame by frame (255 bytes per chunk)
recovered_binary_stream = bytearray()

with open(extracted_framed_file, 'rb') as f_in:
    frame_idx = 0
    while True:
        raw_frame = f_in.read(FRAME_SIZE)
        if not raw_frame:
            break  # End of file reached
        
        # A. Decode and dynamically repair the frame using Reed-Solomon
        try:
            # rs.decode returns a tuple: (decoded_bytes, decoded_with_err_corrected, err_pos)
            decoded_block = rs.decode(raw_frame)[0]
        except ReedSolomonError:
            print(f"CRITICAL ERROR: Frame {frame_idx} is too corrupted to repair!")
            # Clean up and abort safely
            os.remove(extracted_framed_file)
            exit(1)
        
        # B. Slice the decoded 223 bytes into its core components
        nonce = decoded_block[0:16]
        encrypted_payload = decoded_block[16:221]
        received_crc = decoded_block[221:223]
        
        # C. Re-calculate and verify the CRC-16 Checksum
        computed_crc = calculate_crc16(nonce + encrypted_payload)
        if computed_crc != received_crc:
            print(f"CRITICAL ERROR: Integrity validation failed on frame {frame_idx} (Bad CRC).")
            os.remove(extracted_framed_file)
            exit(1)
            
        # D. Decrypt the payload via XOR
        decrypted_payload = xor_encrypt_decrypt(encrypted_payload, XOR_KEY)
        
        # Collect recovered bytes
        recovered_binary_stream.extend(decrypted_payload)
        frame_idx += 1

# 4. Remove internal trailing padding safely (strips extra trailing null bytes)
final_clean_text = recovered_binary_stream.rstrip(b'\x00').decode('utf-8', errors='ignore')

# 5. Write the final cleaned text to disk temporarily so Notepad can read it
with open(TEMP_TEXT_PATH, 'w', encoding='utf-8') as f_out:
    f_out.write(final_clean_text)

# 6. Launch Notepad synchronously and halt execution until the user closes it
print("Opening text contents in Notepad...")
subprocess.run(['notepad.exe', TEMP_TEXT_PATH])

# 7. Post-Processing Cleanup: Wipe text file, intermediate framed files, and pointer files
print("Wiping file traces from memory and storage...")
if os.path.exists(TEMP_TEXT_PATH):
    os.remove(TEMP_TEXT_PATH)
if os.path.exists(extracted_framed_file):
    os.remove(extracted_framed_file)
if os.path.exists(temp_pointer_file):
    os.remove(temp_pointer_file)

print("Execution finished securely.")
