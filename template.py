import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "anime_rec_engine"

# List of files and directories to create
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/config.py",
    f"src/{project_name}/data_models.py",
    f"src/{project_name}/llm_models.py",
    f"src/{project_name}/pipeline.py",
    f"src/{project_name}/prompts.py",
    f"src/{project_name}/recommender.py",
    f"src/{project_name}/vector_store.py",
    "deployment/kubernetes/deployment.yaml",
    "deployment/kubernetes/service.yaml",
    "deployment/Dockerfile",
    "docs/01_gcp_vm_setup.md",
    "docs/02_github_integration.md",
    "docs/03_gcp_firewall_setup.md",
    "docs/04_kubernetes_deployment.md",
    "docs/05_grafana_monitoring.md",
    "main.py",
    "requirements.txt",
    "setup.py",
    ".env.example",
    "README.md"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Create the directory if it doesn't exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    # Create the file if it doesn't exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass  # Create an empty file
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
