import os
from pathlib import Path
import json

raw_dir = Path("raw_dataset")
stats = {}

if raw_dir.exists():
    for class_dir in raw_dir.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*")))
            stats[class_dir.name] = count

print(json.dumps(stats, indent=4))
with open("dataset_stats.json", "w") as f:
    json.dump(stats, f)
