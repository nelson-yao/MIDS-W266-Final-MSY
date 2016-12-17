import nltk.data
import numpy as np
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def load_features(input_data, doc_n=200):
    
    return_data = []
    for i in range(input_data.shape[0]):
        tokens = word_tokenize(input_data.iloc[i])
        tokens = list(set(tokens) - set(stopwords.words('english')))
        
        out = ' '.join(preprocessor(word) for word in tokens[:doc_n])
        return_data.append(out)
        #return_data.append(tokenizer.tokenize(input_data.iloc[i]))
    
    return (return_data) #return_data)

def preprocessor(s):
    """
    """
    s=s.lower().strip()
    s=re.sub("[,.!?:;/~*]"," ",s)
    #remove duplicated 0s and 1s
    s=re.sub("[0-9]*","",s)
    #Number longer than 5 digit
    s=re.sub("[0-9]{5,}","",s)
    #stem end with 'ly'
    s=re.sub("ly\s"," ",s)
    #remove plural
    s=re.sub("s\s"," ",s)
    s=re.sub("s\Z"," ",s)
    #remove _ as start of the word
    s=re.sub("\s[_]+"," ",s)
    #remove stem end with 'ness'
    s=re.sub("ness\s"," ",s)
    s=re.sub("ing\s"," ",s)

    return s


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Create a batch iterator.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]