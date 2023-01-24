# UDParse

UDParse is a fork of UDPipe-Future which in turn is prototype for UDPipe 2.0. The prototype consists of tagging
and parsing and is purely in Python. It participated in CoNLL 2018 UD Shared Task and was one of three winners.
The original code is available at https://github.com/CoNLL-UD-2018/UDPipe-Future.

UDparse has (as UdpipeFuture) the [Mozilla Public License Version 2.0](LICENSE)

UDParse has integrated input sentence vectorisation (with BERT, XLM-Robert-large etc) in order to improve the
quality of the tagged and parsed output.

Compared to the CoNLL 2018 Shared task, UDParse [improves on nearly all treebanks considerably](doc/results.md):

![Comparison XLM-R](doc/conll18_LAS_XLM-R.png)

# Installation

We used anaconda in order to install an virtual environment

```
conda create -n udparse python=3.8
conda activate udparse
conda install cudatoolkit==11.3.1
conda install cudnn==8.2.1
pip --no-cache-dir install tensorflow-gpu==2.5.0
pip --no-cache-dir install tensorflow_addons==0.13.0
pip --no-cache-dir install transformers==4.6.1
pip --no-cache-dir install sentencepiece==0.1.95
pip --no-cache-dir install cython
pip --no-cache-dir install git+https://github.com/andersjo/dependency_decoding 
pip --no-cache-dir install pyyaml
pip --no-cache-dir install matplotlib psutil
pip --no-cache-dir install flask==2.0.1
pip --no-cache-dir install flask-cors==3.0.10
pip --no-cache-dir install flask-restful-swagger-3==0.1
pip --no-cache-dir install flask-restful==0.3.9
pip --no-cache-dir install regex==2021.11.10
pip --no-cache-dir install conllu==4.4.1
pip --no-cache-dir install svgwrite==1.4.2

export LD_LIBRARY_PATH=$HOME/anaconda3/envs/udparse/lib/
```

some transformer models do not exist for Tensorflow. In order to use them you have to install pytorch as well

```
pip --no-cache-dir install pytorch==1.8.1
```

or on more recent GPUs

```
pip3 install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

In order to be able to process raw text, you need to install a [UDPipe 1.2 clone](https://github.com/ioan2/udpipe) tokenizer from
our fork of UDpipe at https://github.com/ioan2/udpipe

```
sudo apt install libpython3.8-dev
git clone https://github.com/ioan2/udpipe

pushd udpipe/src
make
popd

pushd bindings/python
make PYTHON_INCLUDE=/usr/include/python3.8
popd
```

copy UDpipe's `bindings/python/ufal/*` and `bindings/python/ufal_udpipe.so` to to UDParse's `UDParse` or use `PYTHONPATH=...` to specify the location of the UDPipe-bindings in order python can find it

# Train a model (on [Universal Dependencies](https://universaldependencies.org) data)

First prepare a configuration file `data.yml`

```
configs:
  fr-bert:
    calculate_embeddings: bert
    dev: /universal-dependencies/ud-treebanks-v2.8/UD_French-GSD/fr_gsd-ud-dev.conllu
    embeddings: /universal-dependencies/models/2.8/udpf/tmp/data/fr-bert
    gpu: 0
    out: /universal-dependencies/models/2.8/udpf/fr-bert
    test: /universal-dependencies/ud-treebanks-v2.8/UD_French-GSD/fr_gsd-ud-test.conllu
    tokmodel: /universal-dependencies/models/2.8/tok/fr_gsd.tok.model
    train:
    - /universal-dependencies/ud-treebanks-v2.8/UD_French-GSD/fr_gsd-ud-train.conllu
```

* `gpu:` indicates the gpu device starting with 0. If the number given does not correspond to any device or if it is negative
training will be done un CPU.
* `calculate_embeddings` indicates the transformer model to use. The following are recognized:
  * `bert`: bert-base-multilingual-cased
  * `distilbert`: distilbert-base-multilingual-cased
  * `itBERT`: dbmdz/bert-base-italian-xxl-cased
  * `arBERT`: asafaya/bert-base-arabic
  * `fiBERT`: TurkuNLP/bert-base-finnish-cased-v1
  * `slavicBERT`: DeepPavlov/bert-base-bg-cs-pl-ru-cased (needs pytorch)
  * `plBERT`: dkleczek/bert-base-polish-uncased-v1 (needs pytorch)
  * `svBERT`: KB/bert-base-swedish-cased
  * `nlBERT`: wietsedv/bert-base-dutch-cased
  * `flaubert`: flaubert/flaubert_base_cased (needs pytorch)
  * `camembert`: camembert-base 
  * `roberta`: roberta-large
  * `xml-roberta`: jplu/tf-xlm-roberta-large(multilingual)



The tokenizer model (`tokmodel`) must be created using Udpipe 1.2:

```
udpipe --train \
       --tagger=none \
       --parser=none \
       --heldout /universal-dependencies/ud-treebanks-v2.8/UD_French-GSD/fr_gsd-ud-dev.conllu \
      /universal-dependencies/models/2.8/tok/fr_gsd.tok.model \
      /universal-dependencies/ud-treebanks-v2.8/UD_French-GSD/fr_gsd-ud-train.conllu
```

Run the train and test:

```
./run.py --action train --yml data.yml fr-bert
./run.py --action test --yml data.yml fr-bert
```

or in one step:

```
./run.py --action tt --yml data.yml fr-bert
```

The training process puts some debugging stuff into the given `out` directory. To copy only the needed stuff, run:

```
./bin/clonedata.py data.yml new_directory
```

# Use a model

## Predict a file

Tokenise, tag and parse a raw text. If your input file is in CoNLL-U format, omit the `--istext` option:

```
./run.py --action predict \
         --yml data.yml \
         --infile inputtext.txt \
         --istext \
         --outfile output.conllu \
         --istext
```

use the option `--presegmented` if the input file contains one sentence per line. Without this option the input text
will be segmented into sentences before tokenization.

For tokenized (CoNLL-U) files run the following:

```
./run.py --action predict \
         --yml data.yml \
         --infile inputtext.conllu
         --outfile output.conllu
```

For a quick check, whether the model works you can use:

```
./run.py --action predict \
         --yml data.yml \
         --example "A sentence to be tokenized, tagged and parsed"
```

which prints the output (in CoNLL-U format) to stdout


## Server launching

You can launch a server in the following way:


```
./run.py --action server \
    --yml data.yml
    --port 8844 \
    --forcegpu \
    --gpu 1
```

The Swagger API specification can be obtained at http://localhost:**PORT**/api/v1/doc

## Requesting the server

the server can be used with

```
curl -X POST "http://localhost:8844/api/v1/parse" \
    -H "accept: text/tab-separated-values" \
    -H "Content-Type: multipart/form-data" \
    -F "text=my sentence will be parsed" \
    -F "presegmented=false"
```

Or with python's `requests` package

```python
import requests

r = requests.post("http://localhost:8844/api/v1/parse",
                  data = {"text": "my sentence will be parsed",
                          "presegmented": False,
                          "conllu": "", # put tokenized input (CoNLL-U) here instead of in "text". If "conllu" is not empty, "text" will be ignored
                          "parse": True},
                  headers = {"Accept": "text/tab-separated-values" }  # you can set this to get pure CoNLL-U or omot to get json
                 )
print(r.text)
```

## Swagger API documentation

use `http://localhost:8844/api/v1/doc`

## Use programmatically (in python)

load a parser instance

```python
import UDParse.UDParse

udp = UDParse.UDParse(lg=None,
	              action=UDParse.Action.SERVER,
                      yml=<model-directory>,
                      gpu=0,
                      forcegpu=True,
                      usepytorch=True,
                      forceoutdir=False)
while not udp.submitresult.done():
     print("Still loading model ...", model, file=sys.stderr)
     time.sleep(2)
```

use it

```
output = udp.api_process(in_text="my input sentence", is_text=True)
output = udp.api_process(in_text=<string with tokenised sentence in CoNLL-U format>, is_text=False)
```

## Use it as an external dependency library

For the CPU version
```bash
pip install . --find-links=https://download.pytorch.org/whl/torch_stable.html
```

For the GPU version
```bash
pip install .[gpu] --find-links=https://download.pytorch.org/whl/cu113/torch_stable.html
```
