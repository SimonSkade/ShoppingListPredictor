from rscanner.train import trainOnData
from rscanner.scan import scan, loadModel

import sys

debug = False

if len(sys.argv) < 2:
    raise ValueError('Usage: rscanner <train|scan> (image)')

if sys.argv[1] == "train":
    trainOnData()
if sys.argv[1] == "scan":
    if len(sys.argv) < 3:
        raise ValueError('Usage: rscanner <train|scan> (image)')
    if len(sys.argv) == 4:
        if sys.argv[3] == "-d":
            debug = True
    loadModel("rscanner/state_dicts/model.pt")
    print(scan(str(sys.argv[2]),debug))