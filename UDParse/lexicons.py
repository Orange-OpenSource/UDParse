#!/usr/bin/env python3

# loads full form lexicons for postparsing lemma correction

import os
import time
import sys
import json

import marisa_trie
    
class Lexicons:
    def __init__(self, cfg=None):
        self.lgs = {} # LG: FullForm-obj
        if cfg:
            self.loadconfig(cfg)

    def loadconfig(self, cfg):
        # format:
        # lg: file1 file2

        ifp = open(cfg)
        for line in ifp:
            line = line.strip()
            if not line or line[0] == "#":
                continue
            elems = line.split()
            for ix in range(1, len(elems)):
                # TODO add multiple files
                if os.path.isabs(elems[ix]):
                    fn = elems[ix]
                else:
                    fn = os.path.dirname(cfg) + "/" + elems[ix]
                print('Adding full form lexicon "%s" for %s' % (fn, elems[0]), file=sys.stderr)
                self.add(elems[0], fn)

        
    def add(self, lg, lexiconfile):
        self.lgs[lg] = FullForms([lexiconfile])

    def correctlemmas(self, lg, conllu):
        if not lg:
            # detect language
            languages = {} # lf: freq
            cwords = conllu.split("\n")
            for cw in cwords:
                if not cw or cw[0] == "#":
                    continue
                elems = cw.split("\t")
                feats = elems[5].split("|")
                for f in feats:
                    if f.startswith("Lang"):
                        lg = f[5:]
                        if not lg in languages:
                            languages[lg] = 1
                        else:
                            languages[lg] += 1
            if len(languages) > 0:
                lg = sorted(languages.items(), key=lambda x: -x[1])[0][0]
                if lg in self.lgs:
                    return self.lgs[lg].correctlemmas(cwords)
            
            return conllu

        if lg in self.lgs:
            return self.lgs[lg].correctlemmas(conllu)
        else:
            return conllu


class FullForms:
    def __init__(self, lexiconfiles):
        self.forms = {} # form: {upos: [lemmas]}
        self.istrie = False
        self.lemmas = set()

        if lexiconfiles[0].endswith(".compiled"):
            self.loadmarisa(lexiconfiles[0])
        else:
            self.load(lexiconfiles)

    def loadmarisa(self, lexiconfile):
        self.forms = marisa_trie.BytesTrie()
        self.forms.load(lexiconfile)
        self.istrie = True
        print("%s loaded" % lexiconfile)

    def load(self, lexiconfiles):
        #ifp = open("/users/langnat/tmp/johannes/full_form.fr.UD.txt")
        # file format: form lemma UPOS [traits]


        for lexiconfile in lexiconfiles:
            ifp = open(lexiconfile)
            start = time.time()
            for lnr,line in enumerate(ifp, 1):
                elems = line.split()
                #print("LL", lnr, line.strip())
                upos = elems[2]
                if not upos in ["NOUN", "VERB", "ADJ", "ADV"]:
                    continue
                form = elems[0]
                lemma = elems[1]

                self.lemmas.add(lemma)
                if not form in self.forms:
                    self.forms[form] = {}
                if not upos in self.forms[form]:
                    self.forms[form][upos] = []
                self.forms[form][upos].append(lemma)
                if lnr % 100000 == 0:
                    print("%d lines read..." % lnr, end="\r", file=sys.stderr)

            end = time.time()
            ctlemmas = len(self.lemmas)
            ctforms = len(self.forms)
            print("%d forms with %d lemmas (%.2f forms per lemma) loaded in %d secs" % (ctforms, ctlemmas, ctforms/ctlemmas, end-start), file=sys.stderr)

    def getlemma(self, form, lemma, upos):
        if not form in self.forms:
            return None, None
        dico = self.forms[form]
        if self.istrie:
            dico = json.loads(dico[0])
            #print("112ooooo",dico, upos, upos in dico)

        if not upos in dico:
            # no lemma for the current form and the given upos
            # return first lemma for first upos
            upos = next(iter(dico.keys()))
            if upos in ["VERB", "NOUN", "ADV", "ADJ"]:
                return dico[upos][0], upos
            else:
                return None, None
        elif lemma in dico[upos]:
            return lemma, upos
        else:
            # return first lemma for given upos
            return dico[upos][0], upos

#    def check(self, form, lemma):
#        if not form in self.forms:
#            return 0
#        for u,ls in self.forms[form].items():
#            if lemma in ls:
#                return 1
#        return -1

    def correctlemmas(self, conllu):
        newlines = []
        #languages = {} # lg: freq
        if isinstance(conllu, str):
            cwords = conllu.split("\n")
        else:
            cwords = conllu
        corrections = [] # ids of corrected lemmas
        for line in cwords:
            #print("aaaaaaa", line)
            if not line:
                if corrections:
                    newlines.append("# corrected\t" + "\t".join(corrections))
                corrections = []
                newlines.append(line)
            elif line[0] == "#":
                newlines.append(line)
            else:
                elems = line.split("\t")
                
                if elems[3] in ["VERB", "NOUN", "ADJ", "ADV"]:
                    # correct lemma if in lexicon
                    newlemma, newupos = self.getlemma(elems[1], elems[2], elems[3])
                    if newlemma:
                        if elems[2] != newlemma:
                            corrections.append(elems[0])
                            if elems[9] == "_":
                                elems[9] = "OrigLemma=" + elems[2]
                            else:
                                elems[9] += "|OrigLemma=" + elems[2]
                            elems[2] = newlemma
                    if newupos:
                        if elems[3] != newupos:
                            corrections.append(elems[0])
                            if elems[9] == "_":
                                elems[9] = "OrigUPOS=" + elems[3]
                            else:
                                elems[9] += "|OrigUPOS=" + elems[3]
                            elems[3] = newupos
                        
                newlines.append("\t".join(elems))

        return "\n".join(newlines) #, lg

    def compile(self, outfn):
        if not self.istrie:
            output = {}
            for f in self.forms:
                output[f] = bytes(json.dumps(self.forms[f]), "utf8")
            trie = marisa_trie.BytesTrie(zip(output.keys(), output.values()))
            trie.save(outfn + '.marisa')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=str, help="lexicon files")
    parser.add_argument("--compile", "-c", default=None, type=str, help="outfile for compilation into marisa-TRIE")

    parser.print_help()
    args = parser.parse_args()

    if len(args.files) > 0:
        ff = FullForms(args.files)
        if args.compile:
            ff.compile(args.compile)
            

    else:
        # test
        lxs = Lexicons("/users/langnat/tmp/johannes/full_form.cfg")

        #import random
        #lg = "SK"
        #forms = list(lxs.lgs[lg].forms.keys())
        #testforms = []

        #for x in range(50):
        #    ix = random.randint(0, len(forms))
        #    testforms.append(forms[ix])
        #print(testforms)
        #start = time.time()
        #for f in testforms:
        #    print(lxs.lgs[lg].forms[f])
        #print("%.2f secs" % (time.time()-start))

        #while True:
        #    form = input("ENTER to quit ")
        #    if not form:
        #        break
        #    if form in lxs.lgs[lg].forms:
        #        #print(lxs.lgs[lg].forms[form])
        #        print(json.loads(lxs.lgs[lg].forms[form][0]))

        while True:
            form = input("ENTER to quit>> ")
            if not form:
                break

            for lx in lxs.lgs.values():
                if form in lx.forms:
                    print(json.loads(lx.forms[form][0]))
