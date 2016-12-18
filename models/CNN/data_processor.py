import nltk.data
import numpy as np
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def load_features(input_data, doc_n=200):
    
    return_data = []
    for i in range(input_data.shape[0]):
        #tokens = word_tokenize(input_data.iloc[i])
        #tokens = list(set(tokens) - set(stopwords.words('english')))
        #tokens = filter(lambda x: x not in stopwords.words('english'), paragraph.split())
        
        paragraph = input_data.iloc[i].replace('TEXT:', '').replace('Table of Contents', '')\
        .replace("[,.!?:;/~*]", '').replace('Pre-commencement communications pursuant to Rule','')\
        .replace("Exhibits",'').replace("Securities Exchange Act of 1934",'').replace("Securities Act of 1933",'')\
        .replace("Exchange Act",'').replace("8-K",'').replace("1933",'').replace("1934",'')
        tokens = list(set(paragraph.lower().split()) - set(stopwords.words('english')))
        
    #start_index = 20
    #end_index = min(start_index + doc_n, len(tokens))     
        out = ''
        for i, word in enumerate(tokens):
            word = preprocessor(word)
            if word != '' and i > 0:
                out = ' '.join([out, word])
            elif word != '' and i == 0:
                out = ''.join([out, word])
        
        return_data.append(out)
        
        #out = ' '.join(preprocessor(word) for word in tokens[start_index:end_index])
        #return_data.append(tokenizer.tokenize(input_data.iloc[i]))
    return (return_data)


def preprocessor(s):
    """
    Parsing to remove symbols, white space, and comma, etc.
    """
    s=s.lower().strip()
    s=re.sub('[!@#$:;,]', '', s)
    s=re.sub("[,.!?:;/~*()]","",s)
    re.sub(',','', s)
    #remove duplicated 0s and 1s
    #s=re.sub("[0-9]*","",s)
    #Number longer than 5 digit
    #s=re.sub("[0-9]{5,}","",s)
    #s=re.sub("\d*", '', s)
    #s=re.sub("[0-9]{1,}","number",s)
    #stem end with 'ly'
    s=re.sub("ly\s","",s)
    #remove plural
    #s=re.sub("s\s"," ",s)
    #s=re.sub("s\Z"," ",s)
    #remove _ as start of the word
    s=re.sub("\s[_]+","",s)
    #remove stem end with 'ness'
    s=re.sub("ness\s","",s)
    s=re.sub("[|_|]","",s)
    s=re.sub("ing\s","",s)
    s = re.sub('"registrant"',"",s)
    #s=re.sub(r'[^\w]', ' ', s)
    return s


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Create data batches for training
    """
    #np.random.seed(9)
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

def one_iter(data):
    """
    Create a batch iterator for development set
    """
    data = np.array(data)
   
    for batch_num in range(1):
        yield data