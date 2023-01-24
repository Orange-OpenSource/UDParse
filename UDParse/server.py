#!/usr/bin/env python3

# Software Name: UDParse
# SPDX-FileCopyrightText: Copyright (c) 2022 Orange
# SPDX-License-Identifier: Mozilla Public License 2.0
#
# This software is distributed under the MPL-2.0 license.
# the text of which is available at https://www.mozilla.org/en-US/MPL/2.0/
# or see the "LICENSE" file for more details.
#
# Author: Johannes HEINECKE <johannes(dot)heinecke(at)orange(dot)com> et al.



# used start a server

import sys
import time
import os
import socket
import json

from flask import Flask, escape, request, Response, jsonify, render_template
from flask_cors import CORS #, cross_origin

import UDParse.conllu2svg as conllu2svg
import UDParse.version as version

import UDParse.exceptions as udpexceptions


ID = 0
FORM = 1
LEMMA = 2
UPOS = 3
XPOS = 4
FEAT = 5

API_VERSION = "1.0.0"

API_PATH = "/api/v1"
mydir = os.path.dirname(__file__)

class UDParseServer:
    def __init__(self,
                 udparseinstance,
                 port,
                 args,
                 ):
        self.udparse = udparseinstance

        app = Flask("UdpipeFuture")
        app = Flask("UDParse",
                    static_url_path='',
                    static_folder="%s/ui" % mydir,
                    template_folder="%s/ui" % mydir,
                    )

        try:
            ifp = open("%s/api.json" % mydir)
            apidoc = ifp.read()
            ifp.close()
        except:
            apidoc = "{}"

        #print(dir(app))
        CORS(app)

        @app.route("/", methods=["GET"])
        def gui():
            return render_template('index.html') #, toolname=name)

        @app.route(API_PATH + "/doc", methods=["GET"])
        def doc():
            return Response(apidoc, 200, mimetype="application/json")

        
        @app.route('/infoh',methods=["GET"])
        def infoh():
            #print(dir(request))
            return Response(self.udparse.api_infoh(), 200, mimetype="application/json")
            #return " ".join(sys.argv) + "\n"

        @app.route('/info',methods=["GET"])
        def info():
            #print(dir(request))
            #cwd = os.path.abspath(os.path.curdir)
            #hostname = socket.gethostname()
            #return " ".join(sys.argv) + "\n"
            #return Response("%s:%s\n%s\n" % (hostname, cwd, " ".join(sys.argv)), 200 , mimetype="text/plain")
            return Response(self.udparse.api_info(), 200, mimetype="application/json")

        @app.route(API_PATH + '/status',methods=["GET"])
        @app.route('/status',methods=["GET"])
        def status():
            #print(dir(request))
            #return " ".join(sys.argv) + "\n"

            status = "ok"
            rtc = self.hasloaded()
            if rtc:
                status = "loading"
            dico = { "name": "udparse_api",
                     "status": "ok",
                     "version": API_VERSION,
                     "components" : [{
                         "name": "UDParse",
                         "status": status,
                         "version": version.getVersion()
                         }]
                     }
            
            if args.originalinfo:
                dico["data"] = {}
                for k,v in args.originalinfo.items():
                    dico["data"][k] = v
            
            return Response("%s\n" % json.dumps(dico), 200 , mimetype="application/json")


        @app.route('/config',methods=["GET"])
        def config():
            #print(dir(request))
            #return "%s\n" % args
            #return Response("%s\n" % args, 200 , mimetype="text/plain")
            return Response(self.udparse.api_info(), 200, mimetype="application/json")


        @app.route('/updateDebug/<debug>',methods=["GET"])
        def updateDebug(debug):
            return "not implemented\n"

        @app.route(API_PATH + '/tokenize',methods=["GET", "POST"])
        @app.route('/tokenize',methods=["POST", "GET"])
        def tokenise():
            rtc = self.hasloaded() # raises error if data is not yet loaded
            if rtc: return rtc

            text_input = self.checkParameter(request, 'text' , 'string', isOptional=False, defaultValue=None)
            presegmented = self.checkParameter(request, 'presegmented' , 'boolean', isOptional=True, defaultValue=False)

            result = self.udparse.api_process(in_text = text_input,
                                              is_text = True,
                                              is_pre_segmented = presegmented,
                                              tok_only = True,
                                              do_parse = False)
            requestMimeTypes = [request.headers["Accept"].split(";")[0].split(",")[0]]
            if 'text/tab-separated-values' in requestMimeTypes:
                return Response("%s\n" % result, 200 , mimetype="text/tab-separated-values")
            else:
                return Response(json.dumps({"result": result}), 200 , mimetype="application/json")

        


        #@app.route('/',methods=["POST"])
        @app.route(API_PATH + '/parse',methods=["GET", "POST"])
        @app.route('/parse',methods=["POST", "GET"])
        def parse():
            rtc = self.hasloaded() # raises error if data is not yet loaded
            if rtc: return rtc

            text_input = self.checkParameter(request, 'text' , 'string', isOptional=True, defaultValue=None)
            presegmented = self.checkParameter(request, 'presegmented' , 'boolean', isOptional=True, defaultValue=False)
            parse = self.checkParameter(request, 'parse' , 'boolean', isOptional=True, defaultValue=True)
            do_format = self.checkParameter(request, 'format' , 'boolean', isOptional=True, defaultValue=False)
            correctlg = self.checkParameter(request, 'correct' , 'string', isOptional=True, defaultValue=None)
            print("txt: <%s>" % text_input, file=sys.stderr)
            if not text_input:
                conllu_input = self.checkParameter(request, 'conllu' , 'string', isOptional=False, defaultValue=None)

                print("conllu: <%s>" % conllu_input, file=sys.stderr)
            if not text_input:
                # already tokenised
                tok = conllu_input
                if not tok.endswith("\n"):
                    tok += "\n\n"
                elif not tok.endswith("\n\n"):
                    tok += "\n"
                result = self.udparse.api_process(in_text = tok,
                                                  is_text = False,
                                                  is_pre_segmented = presegmented,
                                                  tok_only = False,
                                                  do_parse = parse,
                                                  correct_lg = correctlg)
            else:
                result = self.udparse.api_process(in_text = text_input,
                                                  is_text = True,
                                                  is_pre_segmented = presegmented,
                                                  tok_only = False,
                                                  do_parse = parse,
                                                  correct_lg = correctlg)

            #print("eeee", result)
            if do_format:
                # format for UI display
                allsentences = []
                trees = []
                dico = []
                for sentence in result.strip().split("\n\n"):                        
                    newsent = []
                    corrections = []
                    for line in sentence.split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        if line[0] == "#":
                            if line.startswith("# corrected"):
                                corrections = [int(x) for x in line.split("\t")[1:]]
                            continue
                        newsent.append(line.split("\t"))
                    allsentences.append(newsent)
                    cs = conllu2svg.Conllu2Svg()
                    trees.append(cs.svg(sentence))
                    dico.append({
                        "conllu": allsentences[-1],
                        "tree": trees[-1],
                        "corrections": corrections
                        })
                #print(json.dumps(dico, indent=2))
                return Response(json.dumps(dico), 200 , mimetype="application/json")
            else:
                requestMimeTypes = [request.headers["Accept"].split(";")[0].split(",")[0]]
                if 'text/tab-separated-values' in requestMimeTypes:
                    return Response("%s\n" % result, 200 , mimetype="text/tab-separated-values")
                else:
                    return Response(json.dumps({"result": result}), 200 , mimetype="application/json")



        @app.errorhandler(udpexceptions.UDParseError)
        def handle_invalid_usage(error):
            response = jsonify({"message": error.message}) #jsonify(error.to_dict())
            response.status_code = 400 #error.status_code
            return response



        app.run(host="0.0.0.0", port=port) #, threaded=False, processes=4)


    def hasloaded(self):
        #print("zzzzzzzzzzzz", self.udparse.status, self.udparse.submitresult.done())
        if not self.udparse.submitresult.done():
            err = {"code":3,
                   "description":"The service is not yet available.",
                   "message": "still loading"}

            return Response(json.dumps(err), 503 , mimetype="application/json")
        elif self.udparse.status != "ok":
            err = {"code": 4,
                   "description": "The service is not available.",
                   "message": str(self.udparse.status) }

            return Response(json.dumps(err), 503 , mimetype="application/json")
        else:
            return None


#    def getAttachmentWeights(self, conllu):
#        # try to increase weights for attachments where we are certain that it must be like this
#        # (rule base)
#        return {}
#        # get all non MWE tokens
#        lines = []
#        forms = []
#        for line in conllu.split("\n"):
#            line = line.strip()
#            if line and line[0] != "#":
#                elems = line.split("\t")
#                if not "-" in elems[0]:
#                    lines.append(elems)
#                    forms.append(elems[1])
#
#        sentlen = len(lines)
#        attachments = {} # word ID: correct head ID
#        skipuntil = -1
#        for x in range(sentlen):
#            if x <= skipuntil:
#                continue
#            # if x is the beginning of a chunk of "fixed" Deprels, continue
#            # print("\nXX",x, forms[x:])
#            wordlist = None
#            if self.tree:
#                rtc,wordlist = self.tree.contains(forms[x:], partly=True, debug=0)
#                #print("test subtree", rtc, wordlist)
#                
#                if rtc:
#                    print("forms", forms[x:], "is part of 'fixed' subtree:", wordlist)
#                    # process the fixed dependants
#                    for d in range(1, len(wordlist)):
#                        print(lines[x+d][ID], lines[x+d][FORM], " has head ", lines[x][ID], lines[x][FORM])
#                    #    attachments[int(lines[x+d][ID])] = int(lines[x][ID])
#                    skipuntil = x + d
#                    continue
#
#            #print("apply rules")
#            skipuntil = -1
#            if x < sentlen - 1:
#                if lines[x][UPOS] == "AUX" and lines[x][LEMMA] == "avoir" and \
#                   lines[x+1][XPOS] == "PARTP":
#                    attachments[int(lines[x][ID])] = int(lines[x+1][ID])
#                if lines[x][UPOS] == "DET" and lines[x][LEMMA] in [ "le", "la"] and \
#                   lines[x+1][XPOS] == "NOUN":
#                    attachments[int(lines[x][ID])] = int(lines[x+1][ID])
#                if lines[x][LEMMA] == "qui" and \
#                   lines[x+1][XPOS] == "VERB":
#                    attachments[int(lines[x][ID])] = int(lines[x+1][ID])
#
#
#            if lines[x][UPOS] == "ADP":
#                if x < sentlen - 2:
#                    if lines[x+1][UPOS] in ["NOUN", "PROPN", "VERB", ]:
#                        attachments[int(lines[x][ID])] = int(lines[x+1][ID])
#                    elif lines[x+2][UPOS] == "NOUN" and not lines[x+1][UPOS] in [ "NUM", "VERB" ]:
#                        attachments[int(lines[x][ID])] = int(lines[x+2][ID])
#                    elif lines[x+1][UPOS] in ["PRON", "AUX"]:
#                        if lines[x+1][XPOS] in ["REL"]:
#                            attachments[int(lines[x][ID])] = int(lines[x+1][ID])
#                        elif lines[x+2][UPOS] in ["VERB"]:
#                            attachments[int(lines[x][ID])] = int(lines[x+2][ID])
#                        #else:
#                        #    attachments[int(lines[x][ID])] = int(lines[x+1][ID])
#        print(attachments)
#        return attachments

    # TODO
    def check_mimetype(request, allowedMimeTypesDict):
        """will check request's Accept header towards the dict of [mimetype => 'short type']
            ex: dict = {"application/json": "json", "text/plain": "text"}
        Return: a tuple: the mimetype for the response, the acronym of the mimetype
        """

        if "Accept" not in request.headers:
            """default mime type is application/json"""
            requestMimeTypes = ["application/json"]
        elif request.headers["Accept"] == "*/*":
            requestMimeTypes = ["application/json"]
        else:
            requestMimeTypes = [request.headers["Accept"].split(";")[0].split(",")[0]]

        #print("Accept = " + requestMimeTypes[0], file=sys.stderr)
        for requestMimeType in requestMimeTypes:
            if requestMimeType in allowedMimeTypesDict.keys():
                return (requestMimeType, allowedMimeTypesDict[requestMimeType])

        raise ServerException("invalid mimetyper '%s'" % paramName)
    
    def checkParameter(self, request, paramName, paramType, isOptional, defaultValue):
        """
        check against the 'request' is paramName is present, 
            is so, according to paramType (one of string, boolean, integer, float), 
            is optional and 
            may have a default value
        """  

        # needed for curl -F txt=@file.txt
        if paramName in request.files:
            bstr = request.files[paramName].read()
            return bstr.decode("UTF-8")



        #for k,v in request.values.items():
        #    print("kkk", k,v)
        if not(paramName in request.values):
            if not isOptional:
                #raise apifactory_errors.Error(27)
                raise ServerException("missing mandatory parameter '%s'" % paramName)
                
            else:
                return defaultValue

        value=request.values[paramName].strip()

        #print("nnnn", paramType, value)
        if paramType == "string":
            if len(value) == 0:
                raise ServerException("Parameter '%s' must not be empty." % paramName)
            else:
                return str(value)

        if paramType == "boolean":
            if not( str(value).lower() in ("true", "1", "false", "0")):
                raise ServerException("Parameter '%s' should be a boolean (i.e. one of 'true', 'false', '0', '1')." % paramName)
            else:
                return (value.lower() in ("true", "1"))
        if paramType == "integer":
            if not isInt(value):
                raise ServerException("Parameter '%s' must be an integer." % paramName)
            else:
                return int(value)
        if paramType == "float":
            if not isFloat(value):
                raise ServerException("Parameter '%s' must be a float." % paramName)
            else:
                return float(value)

        raise ServerException("Another stupid error occurred. Invalid paramtype? %s %s" % (paramName, paramType))


class ServerException(udpexceptions.UDParseError):
    def __init__(self, value):
        self.message = value

        super().__init__(self.message)




