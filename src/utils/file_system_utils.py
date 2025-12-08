import os
from pathlib import Path


def get_project_root() -> Path:
    current_file = Path(__file__).resolve()

    # Chercher la racine du projet en remontant
    for parent in current_file.parents:
        # Vérifier la présence de fichiers marqueurs
        if (
            (parent / ".git").exists()
            or (parent / "setup.py").exists()
            or (parent / "pyproject.toml").exists()
            or (parent / "requirements.txt").exists()
        ):
            return parent

    # Si aucun marqueur trouvé, retourner 2 niveaux au-dessus
    return current_file.parents[1]


def get_outputs_path(subfolder: str = None) -> Path:
    project_root = get_project_root()
    outputs_path = project_root / "outputs"

    # Créer le dossier s'il n'existe pas
    outputs_path.mkdir(parents=True, exist_ok=True)

    if subfolder:
        subfolder_path = outputs_path / subfolder
        subfolder_path.mkdir(parents=True, exist_ok=True)
        return subfolder_path

    return outputs_path


def list_files_recursively(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(os.path.join(root, file))
