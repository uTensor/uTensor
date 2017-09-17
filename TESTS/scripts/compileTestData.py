import os
from pathlib import Path


def mkdir(path, dirname):

    outPath = Path(path)
    if not outPath.exists():
        print("invalid path")
        return

    outPath = outPath / dirname

    if not outPath.exists():
        os.makedirs(outPath)
    elif outPath.exists() and not outPath.is_dir():
        print("invalid path")
        return

    return outPath


TEST_DATA_DIR = mkdir(Path.cwd(), "testData")
run_items = Path.cwd().glob("*/*.py")

for item in run_items:
    exec(open(item).read())
