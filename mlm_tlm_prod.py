from os import  getcwd, path
import os
from os import listdir
from os.path import isfile, join

import torch
import time

from flask import Flask, request
from flask_restful import Resource, Api

from src.utils import AttrDict
from src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from src.model.transformer import TransformerModel

from torch.nn.modules.distance import CosineSimilarity
import torch.utils.model_zoo
import numpy as np

from rq import Queue
from worker import conn
from utils import count_words_at_url

from threading import Timer
import urllib 

app = Flask(__name__)
api = Api(app)

model=None
params=None
dico=None
bpe=None
mot=None
        
lechemin = os.path.normpath(os.path.join(getcwd(), './tools/') )

print([x[0] for x in os.walk(lechemin)])

import subprocess

command = "gcc -std=c++11 -pthread -O3 tools/fastBPE/fastBPE/main.cc -IfastBPE -o tools/fastBPE/fast"   
print('executing g++')     
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print("Happens while running")
(output, err) = process.communicate() #now wait plus that you can send commands to process
#This makes the wait possible
p_status = process.wait()

#This will give you the output of the command being executed
print("Command output: ",output)

print('end of g++')  

print([x[0] for x in os.walk(lechemin)])
onlyfiles = [f for f in listdir( os.path.normpath(os.path.join(getcwd(), './tools/fastBPE') ) ) if isfile(join( os.path.normpath(os.path.join(getcwd(), './tools/fastBPE') ) , f))]
print(onlyfiles)

from tools import fastBPE
print(dir(fastBPE))
bpe = fastBPE.fastBPE( path.normpath(path.join(getcwd(), "./codes_xnli_15") ), path.normpath(path.join(getcwd(), "./vocab_xnli_15") )  )

print('bpe ok')

class XLM(Resource):
    def __init__(self):
        super().__init__()
        print('initializing')
        self.model=None
        self.params=None
        self.dico=None
        self.bpe=None
        self.mot = None
    
    def dwnld(self):
        print('start dwnld')
        url = "https://dl.fbaipublicfiles.com/XLM/mlm_tlm_xnli15_1024.pth"
        urllib.request.urlretrieve(url, "mlm_tlm_xnli15_1024.pth")
        print('end dwnld')
        self.model, self.params, self.dico, self.bpe = initialize_model()
        self.mot = 'test ok'
        print('all initialized')
        return self.model, self.params, self.dico, self.bpe, self.mot
    
    def post(self):
        print(request.json)
        print(request.json['sentences'])
        t0 = time.process_time()
        
        sentences = request.json['sentences']
        sentences = [ tuple(sentences[x]) for x in range(len(sentences))]
        print(sentences)
        print('self : ', self.mot)
        print('global : ', mot)
        print("Supported languages 2: %s" % ", ".join(params.lang2id.keys()))
        score = calculate_similarity(sentences, bpe, model, params, dico)
        score= np.array(score.detach().squeeze())
        print(float(score))
        print(time.process_time() - t0, "seconds process time")

        return {'score': float(score), 'time': time.process_time() - t0}
    
    def get(self):
        return {'employees': "got"}



def initialize_model():
    """
    """
    print('launching model')
    
    chemin = getcwd()
    curPath = chemin if "xlm" in chemin else (os.path.join(getcwd(), 'xlm'))
    
    onlyfiles = [f for f in listdir(chemin) if isfile(join(chemin, f))]
    print(onlyfiles)

#     url = "https://dl.fbaipublicfiles.com/XLM/mlm_tlm_xnli15_1024.pth"
 #    urllib.request.urlretrieve(url, "mlm_tlm_xnli15_1024.pth")


    model_path = os.path.normpath(os.path.join(getcwd(), './mlm_tlm_xnli15_1024.pth') )
    reloaded = torch.load(model_path)
    
#     print('allez le model')
#     response = requests.get(url)
#     print('response downloaded')
#     f = io.BytesIO(response.content)
#     reloaded = torch.load(f)
#     print('file downloaded')
    
#    reloaded = Reloaded.serve()
    
    params = AttrDict(reloaded['params'])
    print("Supported languages: %s" % ", ".join(params.lang2id.keys()))
    
    # build dictionary / update parameters
    dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
    params.n_words = len(dico)
    params.bos_index = dico.index(BOS_WORD)
    params.eos_index = dico.index(EOS_WORD)
    params.pad_index = dico.index(PAD_WORD)
    params.unk_index = dico.index(UNK_WORD)
    params.mask_index = dico.index(MASK_WORD)
    
    #params.attention_dropout = 1 #A ENLEVER
    
    # build model / reload weights
    model = TransformerModel(params, dico, True, True)
    model.load_state_dict(reloaded['model'])
    
    bpe = fastBPE.fastBPE(
            path.normpath(path.join(curPath, "./codes_xnli_15") ),
            path.normpath(path.join(curPath, "./vocab_xnli_15") )  )
    print('fin lecture')
    
    return model, params, dico, bpe

def generate_embedding_tensors(sentences, model, params, dico):
    """
    """
    # add </s> sentence delimiters
    sentences = [(('</s> %s </s>' % sent.strip()).split(), lang) for sent, lang in sentences]
    print(sentences)
    bs = len(sentences)
    slen = max([len(sent) for sent, _ in sentences])

    word_ids = torch.LongTensor(slen, bs).fill_(params.pad_index)
    for i in range(len(sentences)):
        sent = torch.LongTensor([dico.index(w) for w in sentences[i][0]])
        word_ids[:len(sent), i] = sent

    lengths = torch.LongTensor([len(sent) for sent, _ in sentences])
    langs = torch.LongTensor([params.lang2id[lang] for _, lang in sentences]).unsqueeze(0).expand(slen, bs)

    tensor = model('fwd', x=word_ids, lengths=lengths, langs=langs, causal=True).contiguous()
    #print(bs, slen, word_ids, lengths, langs, tensor.size())
    
    return tensor, lengths


def calculate_similarity(sentences, bpe, model, params, dico):
    """
    """
    sent_to_bpe = [sent for sent, lan in sentences]
    
    lan_sent = [lan for sent, lan in sentences]
    
    sentences = bpe.apply(sent_to_bpe)
    # len_sent = [len(sent) for sent in sentences]
    sentences = zip(sentences, lan_sent)

    tensor, len_sent = generate_embedding_tensors(sentences, model, params, dico)
    cm = CosineSimilarity(dim=0)
    return cm.forward(tensor[len_sent[0]-1,0], tensor[len_sent[1]-1,1]) 



print('initialized')
#q = Queue(connection=conn)
#
#result = q.enqueue(count_words_at_url, 'http://heroku.com')
#print('resultt', result.get_id())


def hello():
    print("trying dwnld")
    global model
    global params
    global dico
    global bpe
    global mot
    test=XLM()
    model, params, dico, bpe, mot = test.dwnld()
    print('mot :', mot)
    print("ok dwnld")
    
t = Timer(15.0, hello)
t.start() # after 30 seconds, "hello, world" will be printed

print('launching')

#model, params, dico, bpe = initialize_model()
api.add_resource(XLM, '/xlm') # Route_1

print('name : ' , __name__ )
if __name__ == '__main__':
    print('dnas main')
    app.run(port="5002")
















