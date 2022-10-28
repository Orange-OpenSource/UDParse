#!/usr/bin/env python3

# clone data from data.yml into a new directory (for dockerisation)

import sys
import yaml
import shutil
import os


class DataClone:
    def __init__(self, yml, outdir):
        self.alldata = yaml.safe_load(open(yml))
        self.ymlbasedir = os.path.abspath(os.path.dirname(yml))

        newdata = {}

        # get the first entry of data.yml (must be the data.yml from a training output directory
        lg = list(self.alldata["configs"].keys())[0]

        data = self.alldata["configs"][lg]

        #os.makedirs(outdir, exist_ok=True)

        def copyfile_or_list(value):
            if isinstance(value, str):
                if value[0] != "/":
                    value = self.ymlbasedir + "/" + v
                shutil.copy2(value, outdir)
                # newdata[k] = os.path.basename(value)
                return os.path.basename(value)
            else:
                tlist = []
                for f in value:
                    if f[0] != "/":
                        f = self.ymlbasedir + "/" + f
                    shutil.copy2(f, outdir)
                    tlist.append(os.path.basename(f))
                return tlist

        for k, v in data.items():
            # print(k,v)
            if k == "out":
                # clone directory with model and everything a server needs
                if v[0] != "/":
                    v = self.ymlbasedir + "/" + v
                ignfunc = shutil.ignore_patterns("*.npz", "*.conllu", "log", "log.pdf",
                                                 "epoch*.txt*",
                                                 "checkpoint-?????.*",
                                                 "checkpoint-??????.*")
                #shutil.copytree(v, "%s/out" % outdir, ignore=ignfunc)
                shutil.copytree(v, "%s/." % outdir, ignore=ignfunc)
                newdata[k] = "out"
            #elif k in ["tokmodel"]: #"test", "dev", "fixedchunks", "train"]:
            #    newdata[k] = copyfile_or_list(v)
            #else:
            #    newdata[k] = v

        # print(newdata)
        #with open("{}/data.yml".format(outdir), "w") as ofp:
        #    tmp = {"configs": {lg: newdata}}
        #    yaml.safe_dump(tmp, ofp)
        #    ofp.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: clonedata.py data.yml outdir")
    else:
        dc = DataClone(sys.argv[1], sys.argv[2])
