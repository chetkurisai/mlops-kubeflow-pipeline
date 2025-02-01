import shutil

def deploy_model(model_path: str) -> str:
    """Simulate model deployment"""
    destination = "/mnt/models/model.h5"  # Modify as needed
    shutil.copy(model_path, destination)
    return destination
