#!/usr/bin/env python3

# Software Name: UDParse
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: Mozilla Public License 2.0
#
# This software is distributed under the MPL-2.0 license.
# the text of which is available at https://www.mozilla.org/en-US/MPL/2.0/
# or see the "LICENSE" file for more details.
#
# Author: Johannes HEINECKE <johannes(dot)heinecke(at)orange(dot)com> et al.

import sys
import conllu

DOT_OK=True
try:
    import graphviz

except:
    print("no graphviz support installed",file=sys.stderr)
    DOT_OK=False

import svgwrite
import io
#import drawSvg

orangecolors = {
    "charteOrange": "#ff7900",
    "charteOrangeBright": "#ffe5cc", # inofficial
    "charteViolet": "#9164cd",
    "chartePink": "#ffb4e6",
    "charteGreen": "#50be87",
    "charteBlue": "#4bb4e6",
    "charteYellow": "#ffdc00",
    "charteGray": "#8f8f8f",
    "charteYellowBright": "#fff6b6",
    "charteVioletBright": "#d9c2f0",
    "chartePinkBright": "#ffe8f7",
    "charteGreenBright": "#b8ebd6",
    "charteBlueBright": "#b5e8f7",
    "charteGrayBright": "#d6d6d6",
    "charteYellowDark": "#ffb400",
    "charteVioletDark": "#492191",
    "chartePinkDark": "#ff8ad4",
    "charteGreenDark": "#0a6e31",
    "charteBlueDark": "#085ebd",
    "charteGrayDark": "#595959",
    }


class Conllu2Svg:
    def __init__(self):
        pass


    def svg(self, sentence, write=False, left2right=True):
        self.tokens = {} # pos: (height, Token)
        self.heads = {} # pos: (head, deprel)

        if isinstance(sentence, str):
            sentences = conllu.parse(sentence)

            for osent in sentences:
                sent = osent.to_tree()
                self.processtoken(sent)
                break
        else:
            osent = sentence
            sent = osent.to_tree()
            #print(sent.children)
            self.processtoken(sent)

        xfactor = 60
        yfactor = 30
        xskip = .2*xfactor # horizontal space between boxes
        yskip = 2*yfactor # vertical space between boxes
        boxwidth = 1.4
        boxheight = 1.7*yfactor
        
        dwg = svgwrite.Drawing('test.svg', size=None)

        arrow = dwg.marker(id='markerArrow',
                           insert=(8, 5),
                           size=(10, 10),
                           orient='auto') #, markerUnits='strokeWidth')
        #arrow.add(dwg.path(d="M2,2 L4,5 2,8 L12,5 L2,2", fill=orangecolors.get("charteVioletDark")))
        arrow.add(dwg.path(d="M2,2 L3,5 2,8 L10,5 L2,2", fill=orangecolors.get("charteVioletDark")))
        dwg.defs.add(arrow)

        arrowi = dwg.marker(id='markerArrowInv',
                            insert=(4, 5),
                            size=(10, 10),
                            orient='auto') #, markerUnits='strokeWidth')
        #arrowi.add(dwg.path(d="M2,5 L12,2 10,5 L12,8 L2,5", fill=orangecolors.get("charteVioletDark")))
        arrowi.add(dwg.path(d="M2,5 L10,2 9,5 L10,8 L2,5", fill=orangecolors.get("charteVioletDark")))
        dwg.defs.add(arrowi)

        
        maxheight = 0

        # wordnodes
        wordnodes = {} # tid: (xleft, xcenter, width, ytop, ybottom)

        lasttokenid = sorted(self.tokens)[-1]
        monospaced = False

        # calculate word positions
        accumulatedX = 0
        for tid in sorted(self.tokens, reverse=not left2right):
            height,tok = self.tokens[tid]
            tid -= 1 # start with 0
            postid = tid
            if not left2right:
                postid = lasttokenid-1-tid


            xspace = (postid*boxwidth)*xskip
            yspace = (height)*yskip
            if monospaced:
                wordnodes[tid+1] = ((boxwidth*postid)*xfactor + xspace,
                                       (boxwidth*postid)*xfactor + xspace + xfactor*boxwidth/2,
                                       boxwidth*xfactor,
                                       height*yfactor + yspace,
                                       height*yfactor+boxheight + yspace,
                                    )
            else:
                letters = max(len(tok.token["form"]), 6)*10
                wordnodes[tid+1] = (accumulatedX,
                                       accumulatedX + letters/2,
                                       letters,
                                       height*yfactor + yspace,
                                       height*yfactor+boxheight + yspace,
                                    )

                accumulatedX += xskip + letters
                #print(wordnodes[tid+1], letters)

        # draw words
        for tid in sorted(self.tokens):
            height,tok = self.tokens[tid]
            maxheight = max(maxheight, height) # needed to place the bottom line
            tid -= 1 # start with 0
            postid = tid
            if not left2right:
                postid = lasttokenid-1-tid


            xspace = (postid*boxwidth)*xskip
            yspace = (height)*yskip


            if tok.token["upos"] in ["NOUN", "VERB", "ADJ", "ADV"]:
                fillcolor = orangecolors.get("charteGreenBright")
            elif tok.token["upos"] in ["PUNCT", "SYM", "X"]:
                fillcolor = orangecolors.get("charteGrayBright")
            elif tok.token["upos"] in ["PROPN"]:
                fillcolor = orangecolors.get("charteOrangeBright")
            else:
                fillcolor = orangecolors.get("charteBlueBright")


            xleft,xcenter,width,ytop,ybottom = wordnodes[tid+1]
            #xcenter = xleft + xfactor*boxwidth/2
            dwg.add(dwg.rect(insert=(xleft+1,
                                     ytop+1),
                             size=(width, boxheight),
                             rx=.1*xfactor, ry=.6*yfactor,
                             stroke_width=2,
                             stroke="black",
                             fill=fillcolor))
            dwg.add(dwg.text(tok.token["form"],
                             insert=(xcenter, ytop + .5*yfactor),
                             font_family="Lato",
                             text_anchor="middle",
                             font_weight="bold",
                             font_size=14,
                             fill='black'))
            dwg.add(dwg.text(tok.token["upos"],
                             insert=(xcenter, ytop + 1.1*yfactor),
                             font_family="Lato",
                             text_anchor="middle",
                             font_size=14,
                             fill=orangecolors.get("charteBlueDark")))
            dwg.add(dwg.text("%s" % (tid+1),
                             insert=(xcenter, ytop + 1.5*yfactor),
                             font_family="Lato",
                             text_anchor="middle",
                             font_size=10,
                             fill="black"))
        # bottomline
        for tid in sorted(self.tokens):
            height,tok = self.tokens[tid]
            tid -= 1 # start with 0
            postid = tid
            if not left2right:
                postid = lasttokenid-1-tid

            myspace=(maxheight)*yskip
            yspace=(height)*yskip
            xleft,xcenter,width,ytop,ybottom = wordnodes[tid+1]
            #xcenter = xleft + xfactor*boxwidth/2

            dwg.add(dwg.text(tok.token["form"],
                             insert=(xcenter, (maxheight+3.7)*yfactor + myspace),
                             font_family="Lato",
                             text_anchor="middle",
                             font_size=14,
                             fill='black'))
            dwg.add(dwg.text("%s" % (tid+1),
                             insert=(xcenter, (maxheight+4.2)*yfactor + myspace),
                             font_family="Lato",
                             text_anchor="middle",
                             font_size=12,
                             fill='black'))

            dwg.add(dwg.line((xcenter, (maxheight+3)*yfactor + myspace),
                             (xcenter, ybottom),
                             stroke="#bbbbbb",
                             stroke_width=1.5,
                             stroke_dasharray="8,6"
                             ))

        
        dwg['width'] = boxwidth*tid*xfactor + boxwidth*tid*xskip  + boxwidth*xfactor + 3
        dwg['height'] = (maxheight+4.2)*yfactor + myspace + 3
        dwg['viewBox'] = "0 0 %d %d" % (dwg['width'], dwg['height'])

        # deprels
        for dep,(head,deprel) in self.heads.items():
            if head != 0:
                headpos = wordnodes[head]
                hxleft,hxcenter,width,hytop,hybottom = wordnodes[head]
                deppos = wordnodes[dep]
                dxleft,dxcenter,width,dytop,dybottom = wordnodes[dep]
                #print(head, dep, deprel)

                if deprel in ["nsubj","obj","iobj","csubj","ccomp","xcomp"]:
                    # core args
                    color = orangecolors.get("charteGreenDark")
                elif deprel in [ "obl","vocative","expl","dislocated","advcl","advmod","discourse","aux","cop","mark"]:
                    # noncore args
                    color = orangecolors.get("charteBlueDark")

                elif deprel in ["nmod","appos","nummod","acl","amod","det","clf","case"]:
                    # nominal deps
                    color = orangecolors.get("charteVioletDark")
                elif deprel.startswith("nsubj:"):
                    # core args
                    color = orangecolors.get("charteGreenDark")
                elif deprel.startswith("aux:"):
                    color = orangecolors.get("charteBlueDark")
                else:
                    color = orangecolors.get("chartePinkDark")

                if deppos[0] > headpos[0]:
                    #print(headpos, boxwidth)
                    deprelpath = dwg.path(("M %f %f L %f %f" % (hxcenter, hybottom,
                                                                dxcenter, dytop)),
                                          stroke=color, #orangecolors.get("charteVioletDark"),
                                          stroke_width=2,
                                          style="marker-end: url(#markerArrow);"
                                          )
                else:
                    deprelpath = dwg.path(("M %f %f L %f %f" % (dxcenter, dytop,
                                                                hxcenter, hybottom)),
                                          stroke=color, #orangecolors.get("charteVioletDark"),
                                          stroke_width=2,
                                          style="marker-start: url(#markerArrowInv);"
                                          )

                dwg.add(deprelpath)
                text = dwg.add(dwg.text("", dy=["-4"]))



                text.add(dwg.textPath(path=deprelpath, text=deprel,
                                                startOffset="50%", method='align', spacing='exact',
                                                font_family="Lato",
                                                font_size="14",
                                                text_anchor="middle",
                                                fill=color, #orangecolors.get("charteVioletDark"),
                                                alignment_baseline="central"
                                                ))

        # MWT's
        for token in osent:
            if isinstance(token["id"], tuple):
                #tid = token["id"][0]-1
                #length=token["id"][2]-token["id"][0]+1
                #print(token["id"], token["form"], length )
                #postid = tid
                #if not left2right:
                #    postid = lasttokenid-tid-length
                #print(tid, postid)
                #xspace=(boxwidth*(postid))*xskip

                sxleft,sxcenter,swidth,sytop,sybottom = wordnodes[token["id"][0]]
                exleft,excenter,ewidth,eytop,eybottom = wordnodes[token["id"][2]]
                
                if not left2right:
                    sxleft,sxcenter,swidth,sytop,sybottom = wordnodes[token["id"][2]]
                    exleft,excenter,ewidth,eytop,eybottom = wordnodes[token["id"][0]]

                width = exleft+ewidth-sxleft

                dwg.add(dwg.rect(insert=(sxleft, (maxheight+2)*yfactor + myspace),
                                 size=(width , boxheight/2),
                                 stroke_width=2,
                                 rx=.05*xfactor, ry=.6*yfactor,
                                 stroke="black", #orangecolors.get("charteGrayBright"),
                                 fill=orangecolors.get("charteGrayBright")))
                dwg.add(dwg.text(token["form"],
                                 insert=(sxleft+width/2,
                                         (maxheight+2.6)*yfactor + myspace),
                                 font_family="Lato",
                                 text_anchor="middle",
                                 font_size=14,
                                 fill='black'))

        if write:
            dwg.save(pretty=True, indent=4)
        else:
            s = io.StringIO()
            dwg.write(s, pretty=True, indent=4)
            return s.getvalue()





    def dot(self, sentence, format="svg", write=False):
        if not DOT_OK:
            return ""

        sentences = conllu.parse(sentence)
        self.tokens = {} # pos: (height, Token)
        self.heads = {} # pos: (head, deprel)
        #print(sentences)
        for osent in sentences:
            #print(osent.serialize())
            sent = osent.to_tree()
            #print(sent.children)
            self.processtoken(sent)
            #print(sent.serialize())
            break
        #print(self.tokens)

        dot = graphviz.Digraph('satz',
                               engine='neato',
                               node_attr = {"shape":"box","width":"0.2","height":"0.2","fontname":"Lato","fontsize":"12.0","penwidth":"0.5"},
                               edge_attr = {"fontname":"Lato","fontsize":"12.0","penwidth":"0.5"},
                               format=format)
        


        maxheight = 0
        # word nodes
        for tid in sorted(self.tokens):
            height,tok = self.tokens[tid]
            maxheight = max(maxheight, height)
            if tok.token["upos"] in ["NOUN", "PROPN", "VERB", "ADJ", "ADV"]:
                fillcolor = orangecolors.get("charteGreenBright")
            elif tok.token["upos"] in ["PUNCT", "SYM", "X"]:
                fillcolor = orangecolors.get("charteGrayBright")
            else:
                fillcolor = orangecolors.get("charteBlueBright")
            dot.node("n%s" % tid, label="<%s<BR/><B>%s</B>>" % (tok.token["form"], tok.token["upos"]),
                     fillcolor=fillcolor,
                     style="filled",
                     pos="%d, -%d!" % (tid,height))
        # bottom line
        for tid in sorted(self.tokens):
            height,tok = self.tokens[tid]
            dot.node("b%s" % tid, label="%s\\n%d" % (tok.token["form"], tid),
                     shape="none",
                     pos="%d, -%f!" % (tid, maxheight + .9))
            dot.edge("n%d" % tid, "b%d" % tid, arrowhead="none", color="gray")

        # deprel
        for dep,(head,deprel) in self.heads.items():
            if head != 0:
                dot.edge("n%d" % head, "n%d" % dep, label=deprel,
                         fontcolor=orangecolors.get("charteVioletDark"),
                         color=orangecolors.get("charteVioletDark"))

        # MWT's
        for token in osent:
            if isinstance(token["id"], tuple):
                #print(token["id"])
                length=token["id"][2]-token["id"][0]+1
                pos = (token["id"][2]+token["id"][0])/2
                dot.node("mtw%s" % token["id"][0], label=token["form"],
                         shape="box",
                         width="%d" % length,
                         fillcolor="#dddddd",
                         color="#dddddd",
                         style="filled",
                         pos="%f, -%f!" % (pos, maxheight + .5))

        if write:
            dot.render()
        else:
            return dot.pipe(encoding="utf-8")
        
    def processtoken(self, token, level=0):
        self.tokens[token.token["id"]] = (level, token)
        for ch in token.children:
            self.heads[ch.token["id"]] = (token.token["id"], ch.token["deprel"])
            self.processtoken(ch, level=level+1)

        
if __name__ == "__main__":
    if len(sys.argv)  < 3:
        sentence1 = '''1	Aviator	Aviator	PROPN	_	_	0	root	_	SpaceAfter=No
2	,	,	PUNCT	_	_	1	punct	_	_
3	un	un	DET	_	Definite=Ind|Gender=Masc|Number=Sing|PronType=Art	4	det	_	_
4	film	film	NOUN	_	Gender=Masc|Number=Sing	1	appos	_	_
5	sur	sur	ADP	_	_	7	case	_	_
6	la	le	DET	_	Definite=Def|Gender=Fem|Number=Sing|PronType=Art	7	det	_	_
7	vie	vie	NOUN	_	Gender=Fem|Number=Sing	4	nmod	_	_
8-9	du	_	_	_	_	_	_	_	_
8	de	de	ADP	_	_	10	case	_	_
9	le	le	ADP	_	_	10	case	_	_
10	Howard	Howard	PROPN	_	_	7	nmod	_	_
11	Huuuuuuuuuuuughes	Hughes	PROPN	_	_	10	flat:name	_	SpaceAfter=No
12	.	.	PUNCT	_	_	1	punct	_	_

'''

        sentence2 = '''1	Les	le	DET	_	Definite=Def|Gender=Fem|Number=Plur|PronType=Art	2	det	_	_
2	études	étude	NOUN	_	Gender=Fem|Number=Plur	3	nsubj	_	_
3	durent	durer	VERB	_	Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin	0	root	_	_
4	six	six	NUM	_	_	5	nummod	_	_
5	ans	an	NOUN	_	Gender=Masc|Number=Plur	3	obj	_	_
6	mais	mais	CCONJ	_	_	9	cc	_	_
7	leur	son	DET	_	Gender=Masc|Number=Sing|PronType=Prs	8	nmod:poss	_	_
8	contenu	contenu	NOUN	_	Gender=Masc|Number=Sing	9	nsubj	_	_
9	diffère	différer	VERB	_	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	3	conj	_	_
10	donc	donc	ADV	_	_	9	advmod	_	_
11	selon	selon	ADP	_	_	13	case	_	_
12	les	le	DET	_	Definite=Def|Number=Plur|PronType=Art	13	det	_	_
13	Facultés	Facultés	PROPN	_	_	9	obl	_	SpaceAfter=No
14	.	.	PUNCT	_	_	3	punct	_	_

'''

        cs = Conllu2Svg()
        #cs.dot(sentence, format="pdf", write=True)
        cs.svg(sentence1, write=True)
    else:
        ifp = open(sys.argv[1])
        data = ifp.read()
        sentences = conllu.parse(data)
        nr = int(sys.argv[2])
        cs = Conllu2Svg()
        left2right = True
        if len(sys.argv) > 3:
            left2right = False
        cs.svg(sentences[nr], write=True, left2right=left2right)

        line = input("sentnr> ")
        while line:
            if line == "+":
                nr += 1
            elif line == "-":
                nr -= 1
            else:
                nr = int(line)
            cs.svg(sentences[nr], write=True, left2right=left2right)
            line = input("sentnr> ")

