#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tonyparker
Credits to [1] G. Lample *, A. Conneau * [*Cross-lingual Language Model Pretraining*](https://arxiv.org/abs/1901.07291)
"""
import os
import torch
import numpy.matlib 
import numpy as np 

from src.utils import AttrDict
from src.data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from src.model.transformer import TransformerModel

import subprocess
from torch.nn.modules.distance import CosineSimilarity
cm = CosineSimilarity(dim=0)

#On initialise le modèle
model_path = './mlm_tlm_xnli15_1024.pth'
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

# build model / reload weights
model = TransformerModel(params, dico, True, True)
model.load_state_dict(reloaded['model'])

local=False
if os.environ["TERM_PROGRAM"] == "Apple_Terminal":
    local=True
    
#On crée les fonctions intermédiaires
def generate_embedding_tensors(sentences, cache=None):
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

    tensor = model('fwd', x=word_ids, 
                   lengths=lengths, 
                   langs=langs, 
                   causal=True, 
                   cache=cache).contiguous()
    
    return tensor, lengths

def generate_tensor(sentences, cache=None):
#    sent_to_bpe = [sent for sent, lan in sentences]
    lan_sent = [lan for sent, lan in sentences]
    
    #On le fait ligne par ligne vu que la langue peut changer à chaque fois, pas optimal mais le plus simple
    newSentences = []
    for sent, lan in sentences:
        file = open("input_file","w", encoding="utf-8") 
        file.write( sent + '\n' )
        file.close() 
        
#        print('tokenizing and lowercasing data')    
        process = subprocess.Popen("cat input_file | tools/tokenize.sh " + lan + " | python tools/lowercase_and_remove_accent.py > prep_input", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output, err) = process.communicate() #now wait plus that you can send commands to process
        process.wait()
#        print("Command tokenize output: ",output)
        
#        print('executing bpe')    
        process = subprocess.Popen("./tools/fastBPE" + ("_local" if local else "") + "/fast applybpe output_file prep_input codes_xnli_15", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (output, err) = process.communicate()
        process.wait()
#        print("Command fast output: ",output)
        
#        print("reading file")
        f = open("output_file", "r")
        newSentence= f.read().rstrip('\n')
        f.close()
        newSentences.append(newSentence)
    
#    sentences = bpe.apply(sent_to_bpe)
    sentences = zip(newSentences, lan_sent)
    tensor, len_sent = generate_embedding_tensors(sentences)
    
    return tensor, len_sent

def calculate_similarity(sentences, cache=None):
    """
    """
    tensor, len_sent = generate_tensor(sentences)
    
    cm = CosineSimilarity(dim=0)
    return cm.forward(tensor[len_sent[0]-1,0], tensor[len_sent[1]-1,1]), tensor, len_sent

#On lance le test sur un ensemble de phrases
sentences = [
        ("The technology is there to do it .", 'en'), 
        ("La technologie est là pour le faire .", 'fr'), 
        ("Je veux vivre le jeu .", 'fr'),
        ("Ceci est une phrase complètement aléatoire", 'fr')]
score, t, len_sent = calculate_similarity(sentences)
for i in range(len(sentences)):
    print(cm.forward(t[len_sent[0]-1,0], t[len_sent[i]-1,i]) )
