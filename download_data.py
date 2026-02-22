import kagglehub
import shutil
import os

# Download dataset
path = kagglehub.dataset_download("uthaya1995/90-day-habit-tracker-for-personal-growth")
print("Downloaded to:", path)

# Copy CSV to your project's data/raw folder
for file in os.listdir(path):
    if file.endswith(".csv"):
        src = os.path.join(path, file)
        dst = os.path.join("data/raw", file)
        shutil.copy(src, dst)
        print(f"Copied: {file} → data/raw/{file}")
