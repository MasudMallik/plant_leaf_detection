import os
from pathlib import Path

files=[
    "backend/model_train.ipynb",
    "backend/data_ingestion.py",
    "backend/main.py",
    "backend/DOCKERFILE",
    "frontend/templates/index.html",
    "frontend/DOCKERFILE",
    "backend/logger.py"
]

for file in files:
    filepath=Path(file)

    file_dir,file_path=os.path.split(filepath)

    if file_dir !="":
        os.makedirs(file_dir,exist_ok=True)
        print("Folder created succesfully")
    if not os.path.exists(filepath) or os.path.getsize(filepath)==0:
        with open(filepath,"w") as f:
            pass
         