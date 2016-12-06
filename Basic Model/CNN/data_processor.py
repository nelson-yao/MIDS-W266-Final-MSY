import nltk.data
import numpy as np
from nltk import word_tokenize
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def load_features(input_data):

    return_data = []
    for i in range(input_data.shape[0]):
        tokens = word_tokenize(input_data.iloc[i])
        out = ' '.join(word for word in tokens[:200])
        
        return_data.append(out)
        #return_data.append(tokenizer.tokenize(input_data.iloc[i]))
    
    return (return_data) #return_data)

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