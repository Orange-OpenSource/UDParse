#!/usr/bin/env python3

# client to send CoNLL-U files to UDParse server

import sys
import time
import conllu
import requests


proxies = {
    "http": None,
    "https": None,
}


class Client:
    def __init__(self, host, port, conllufile=None, textfile=None, onlypos=False):
        self.url = "http://%s:%s/api/v1/parse" % (host, port)
        self.ct = 0
        self.parse = not onlypos

        #self.cdoc = ConllParser.ConllDocSentencewise(conllufile)

        self.ifp = open(conllufile)


    def process(self):
        start = time.time()
        self.tokens = 0
        for sentence in conllu.parse_incr(self.ifp):
            self.ct += 1
            
            self.tokens += len(sentence)
            r = requests.post(self.url,
                              data={"conllu": sentence.serialize(),
                                    "parse": self.parse },
                              #headers={"Accept": "application/json"}, 
                              headers={"Accept": "text/tab-separated-values"}, 
                              proxies=proxies)
            print(r.text)


        end = time.time()
        delta = end - start
        print("%d sentences/%d tokens in %.1f seconds (%.2f sentences/sec, %.2f tokens/sec)" % (self.ct, self.tokens,
                                   delta, self.ct/delta, self.tokens/delta), file=sys.stderr)


if __name__ == "__main__":
    # if len(sys.argv) < 4:
    #     print("usage: $0 server port conllufile", file=sys.stderr)
    # else:
    #     cc = Client(sys.argv[1], sys.argv[2], sys.argv[3])
    #     cc.process()

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--onlypos", "-P", default=False, action="store_true", help="only POS tagging")
    parser.add_argument("--server", "-s", required=True, type=str, help="server name")
    parser.add_argument("--port", "-p", required=True, type=int, help="port")
    parser.add_argument("--conllufile", "-c", type=str, help="input file")
    #parser.add_argument("--textfile", "-t",  type=str, help="input file")

    if len(sys.argv) < 2:
        parser.print_help()
    else:
        args = parser.parse_args()

        if not args.conllufile: # and not args.testfile:
            print("input file required", file=sys.stderr)
        else:

            cc = Client(host=args.server, port=args.port, conllufile=args.conllufile, onlypos=args.onlypos)
            cc.process()
