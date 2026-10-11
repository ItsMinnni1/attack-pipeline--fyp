import os
import tempfile
from pathlib import Path

def collect_extract_and_view(target_directory, extension="*.txt"):
    search_path = Path(target_directory)
    file_collection = list(search_path.rglob(extension))
    
    # 1. Create a safe temporary file on the local drive
    temp_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False, encoding='utf-8')
    temp_path = temp_file.name
    
    try:
        for file in file_collection:
            try:
                # 2. Extract contents from the text file
                content = file.read_text(encoding='utf-8')
                
                # 3. Write contents to the temporary file
                temp_file.write(f"=== SOURCE: {file.name} ===\n")
                temp_file.write(content)
                temp_file.write("\n\n")
                print(f"Successfully extracted: {file.name}")
            except Exception as e:
                temp_file.write(f"--- Error reading {file.name}: {e} ---\n\n")
                
    finally:
        # 4. Flush and close the file stream so Windows unlocks it
        temp_file.close()

    # 5. Launch Notepad using the 'temp_path' variable
    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
        print(f"\nOpening temporary file in Notepad...")
        os.system(f'notepad.exe "{temp_path}"')
        
        # 6. Clean up the temp file after you close Notepad
        os.remove(temp_path)
        print("Temporary file cleaned up successfully.")
    else:
        print("\nNo files found or file was empty. Notepad was not opened.")

# Execute using your correct target folder
collect_extract_and_view(r"M:\supersecretfolder\secretsubfolder", "*.txt")
