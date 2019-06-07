from os import path, getcwd
import torch
import time
from six.moves import urllib

from flask import Flask, request
from flask_restful import Resource, Api

from src.utils import AttrDict
from src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from src.model.transformer import TransformerModel

from torch.nn.modules.distance import CosineSimilarity
import fastBPE
import numpy as np

app = Flask(__name__)
api = Api(app)

class XLM(Resource):
    def __init__(self):
        super().__init__()
        print('initializing')
    
    def post(self):
        print(request.json)
        print(request.json['sentences'])
        t0 = time.process_time()
        
        sentences = request.json['sentences']
        sentences = [ tuple(sentences[x]) for x in range(len(sentences))]
        print(sentences)
        
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
    
    print('downloading model')
    url = "https://dl.fbaipublicfiles.com/XLM/mlm_tlm_xnli15_1024.pth"
    urllib.request.urlretrieve(url, "mlm_tlm_xnli15_1024.pth")
    print('file downloaded')
    
    chemin = getcwd()
    curPath = chemin if "xlm" in chemin else (path.join(getcwd(), 'xlm'))

    model_path = path.normpath(path.join(curPath, './mlm_tlm_xnli15_1024.pth') )
    reloaded = torch.load(model_path)
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



model, params, dico, bpe = initialize_model()

print('initialized')

api.add_resource(XLM, '/xlm') # Route_1

if __name__ == '__main__':
     app.run(port='5002')
















