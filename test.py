import os
import tempfile
import zipfile

# 1. Search for the .txt file in subdirectories
source_dir = "M:\supersecretfolder\secretsubfolder"
target_file_path = None

for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith(".txt"):  # Finds your text file
            target_file_path = os.path.join(root, file)
            break
    if target_file_path:
        break

if target_file_path:
    # 2. Compress the original file into a .zip file in the same folder
    zip_file_path = target_file_path.replace(".txt", ".zip")
    
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # arcname saves just the file name inside the zip, avoiding deep nested folders
        zipf.write(target_file_path, arcname=os.path.basename(target_file_path))
    
    print(f"Successfully compressed original file to: {zip_file_path}")

    # OPTIONAL: If you want to delete the uncompressed original .txt right now, uncomment below:
    # os.remove(target_file_path)

    # 3. Create the temporary file containing the PATH to the new zip file
    # (Using delete=False ensures it persists until Script 2 looks for it)
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt')
    temp_file.write(zip_file_path)
    temp_file.close()

    print(f"Temp file created at: {temp_file.name}")
else:
    print("No .txt file found.")
