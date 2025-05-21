import os
import sys

try:
    import gdown
except ImportError:
    print('gdown not found, installing...')
    os.system(f'{sys.executable} -m pip install gdown')
    import gdown

GOOGLE_DRIVE_URL = 'https://drive.google.com/uc?id=1c8oQn6e6vG2W9xZy-6oK7vQjKfJkRk7e'
MODEL_PATH = 'yolov8n-hand.pt'

print(f"Downloading YOLOv8 hand detection model from {GOOGLE_DRIVE_URL} ...")
gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False)
print(f"Model downloaded and saved as {MODEL_PATH}")
