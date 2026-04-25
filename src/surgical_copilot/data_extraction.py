from pathlib import Path
from PIL import Image
from tqdm import tqdm
from zipfile import ZipFile
import argparse

def extraction():

    parser = argparse.ArgumentParser(description="Extract zip files from source to target directory")
    parser.add_argument("--source_dir", type=str, required=True, help="Directory containing the zip files")
    parser.add_argument("--target_dir", type=str, required=True, help="Directory where the extracted files will be stored")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    target_dir = Path(args.target_dir).resolve()
    
    target_dir.mkdir(parents=True, exist_ok=True)
        
    subject_list = sorted([d.name for d in source_dir.iterdir() if d.is_dir()])

    for subject in tqdm(subject_list, desc="Processing subjects"):
        zip_files = list((source_dir / subject).glob("*.zip"))
        
        for zip_file in zip_files:
            with ZipFile(zip_file, 'r') as zip_ref:
                # Otteniamo la lista dei file all'interno dello zip
                for member in zip_ref.infolist():
                    # Definiamo la destinazione in base al contenuto del path nel file zip
                    if "imgs/" in member.filename:
                        dest = target_dir / subject / "imgs"
                    elif "labels/" in member.filename:
                        dest = target_dir / subject / "labels"
                    else:
                        continue 
                    
                    zip_ref.extract(member, dest)

if __name__ == "__main__":
    extraction()
