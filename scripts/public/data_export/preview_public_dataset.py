import datasets
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent.parent

citationSet = datasets.load_dataset("pandas", data_files=str(ROOT_DIR / "data" / "samplesPublic.pkl"))
citationSet.save_to_disk(str(ROOT_DIR / "data" / "samplesPublic"))
citationSet = datasets.load_from_disk(str(ROOT_DIR / "data" / "samplesPublic"))

len(citationSet['train'].unique('citingID'))
