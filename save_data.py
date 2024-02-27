import pandas as pd
from model.dictionary_promptprotein import Alphabet
import lmdb
import json
from utils.conventer import PromptConverter
from poprogress import simple_progress
import mindspore as ms
prompts = ['<seq>']
import pickle
import numpy as np

def save(path):
    ms.set_context(device_target="CPU", mode=ms.PYNATIVE_MODE)
    dictionary = Alphabet.build_alphabet()
    converter = PromptConverter(dictionary)
    data = lmdb.open(path, create=False, subdir=True, readonly=True, lock=False)
    txn = data.begin(write=False)
    # df = pd.DataFrame(columns=['data', 'label'])
    # df.to_csv("./data/train3.csv", index=False)
    list = []
    i =0
    for k, v in simple_progress(txn.cursor() ,desc='converter processing'):
        v = v.decode('utf-8', "ignore")
        origin_tokens, masked_tokens = converter(v, prompt_toks=prompts)
        #.data.append((encoded_sequences[i], self.value[i]))
        #print(encoded_sequence)
        origin_tokens = origin_tokens.asnumpy()
        masked_tokens = masked_tokens.asnumpy()
        list_data = [masked_tokens, origin_tokens]
        np.save('./data/nptest4/'+"%d.npy"%(i), list_data, allow_pickle=True)
        i= i+1
        if i % 10000 == 0:
            print(i)

if __name__ == '__main__':
    save('./uniref50/test')