import os
import shutil

# Get the script file path
script_path = os.path.abspath(__file__)

# Get the directory containing the script file
script_directory = os.path.dirname(script_path)

# Set the source and destination folder paths
source_folder_prefix = "data/batch_"
source_folder_range = range(1, 16)
destination_folder = "data/images"

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)
# Delete all files in the destination folder
existing_files = os.listdir(destination_folder)
for file_name in existing_files:
    file_path = os.path.join(destination_folder, file_name)
    os.remove(file_path)

# Initialize the file counter
file_counter = 0

# Iterate over each source folder
for folder_number in source_folder_range:
    source_folder = os.path.join(script_directory, f"{source_folder_prefix}{folder_number}")

    # Check if the source folder exists
    if os.path.exists(source_folder) and os.path.isdir(source_folder):
        # Get the list of files in the source folder
        file_list = os.listdir(source_folder)

        # Iterate over each file in the source folder
        for file_name in sorted(file_list):
            if file_name.endswith(".jpg"):
                # Get the source file path
                source_file = os.path.join(source_folder, file_name)

                # Construct the destination file path
                destination_file = os.path.join(destination_folder, f"{file_counter:06d}.jpg")
                # destination_file = os.path.join(destination_folder, f"batch_{folder_number}{file_name}")
                # Copy the file to the destination folder
                shutil.copy2(source_file, destination_file)

                # Increment the file counter
                file_counter += 1

print(f"{file_counter} files copied successfully.")
