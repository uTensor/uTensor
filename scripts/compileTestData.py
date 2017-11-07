import os
from pathlib import Path
import shutil 

DATA_DIR = "PRE-GEN"
OUTPUT_DIR = "testData"



def mkdir(path, dirname):

    outPath = Path(path)
    if not outPath.exists():
        print("invalid path")
        return

    outPath = outPath / dirname

    if not outPath.exists():
        os.makedirs(str(outPath))
    elif outPath.exists() and not outPath.is_dir():
        print("invalid path")
        return

    return outPath


mkdir(Path.cwd(), OUTPUT_DIR)
FULL_OUTPUT_DIR = (Path.cwd() / OUTPUT_DIR).resolve()
FULL_PREGEN_DIR = (Path.cwd() / DATA_DIR).resolve()
TEST_DATA_DIR = FULL_OUTPUT_DIR

print("FULL_OUTPUT_DIR", FULL_OUTPUT_DIR)
print("FULL_PREGEN_DIR", FULL_PREGEN_DIR)


#remove the output directry and start clean:
if Path(FULL_OUTPUT_DIR).exists():
    shutil.rmtree(str(FULL_OUTPUT_DIR))

#copy the pre-generated date to output:
shutil.copytree(str(FULL_PREGEN_DIR), str(FULL_OUTPUT_DIR))

#run all scripts to generate the data
run_items = Path.cwd().glob("GEN_SCR/*/*.py")

for item in run_items:
    exec(open(str(item), "r").read())
