import os
import shutil

def sort_files_by_last_digit(input_folder):

    folder_dr = os.path.join(input_folder, "DR") # Output folders
    folder_non = os.path.join(input_folder, "NonDR")

    os.makedirs(folder_dr, exist_ok=True) # Create folders if they don't exist
    os.makedirs(folder_non, exist_ok=True)

    for filename in os.listdir(input_folder):    # Iterate through all files in the directory
        filepath = os.path.join(input_folder, filename)

        if os.path.isdir(filepath):  # Skip directories
            continue

        name_no_ext, _ = os.path.splitext(filename) # Get the name without extension

        # Skip if last char isn't a digit
        if not name_no_ext[-1].isdigit():
            print(f"Skipping (no digit): {filename}")
            continue

        last_digit = int(name_no_ext[-1])

        if last_digit == 1 or last_digit == 2: # Remove labels not needed
            os.remove(filepath)
            print(f"Removed {filename}")
            continue

        if last_digit == 3 or last_digit == 4:          # Decide target folder
            target_file = folder_dr
        else:
            target_file = folder_non

        shutil.move(filepath, os.path.join(target_file, filename))  # Move file
        print(f"Moved {filename} → {target_file}")

if __name__ == "__main__":
    input_path = r"C:\Users\kyguy\Workspace\DeepLearning\DS_IDRID\Train"  # <- replace with your folder path
    sort_files_by_last_digit(input_path)
