##  WORD EMBEDDINGS - GloVe
#  Generates a word embedding based on the pretrained corpora
#     Nov, 2016


import numpy as np

from os.path import abspath
from glob import glob


# Obtain the absolute path to the corpora folder
__path = abspath('.')
__pos = __path.index('MIDS-W266-Final-MSY')
PATH_TO_CORPORA = __path[:__pos + 19] + '/data/geom_corpora/'  # len('MIDS-W266-Final-MSY') --> 19


def list_corpora():
    """Lists the corpora available"""
    non_zips = filter( lambda x: not (x.endswith('.zip') or x.endswith('.gz') or x.endswith('.md')), glob(PATH_TO_CORPORA + '*') )
    return map(lambda x: x.split('/')[-1], non_zips)


class GloVe(object):
    """Trained object of `GloVe<http://nlp.stanford.edu/projects/glove/>`_ using the `stanfordnlp-trained datasets<http://nlp.stanford.edu/data/wordvecs/>`_"""

    PATH_TO_CORPORA = PATH_TO_CORPORA

    def __init__(self, file_name):
        self.vocab = self.load_model(self.PATH_TO_CORPORA + file_name)


    def load_model(self, file_name):
        """Loads a given GloVe model"""
        repr_dict = {}  # initialize dictionary

        with open(file_name, 'rb') as f:
            for line in iter(f.readline, ''):
                _terms = line.split()
                repr_dict[_terms[0]] = np.array(map(float, _terms[1:]))

        return repr_dict

    def __getitem__(self, item):
        """Geometric representation of the object"""
        try:
            representation = self.vocab[item]
        except KeyError:
            representation = self.vocab['<unk>']
        finally:
            return representation


    # NOTE: research other aggregation methods (other than mean)
    def __call__(self, message_text):
        """Geometric representation of a document"""
        representation = 0.
        words = message_text.lower().split()
        if words:
            for w in words:
                representation += self.__getitem__( w.strip() )

            representation /= len(words)
        # If no words were specified
        else:
            representation = self.vocab['<unk>']

        return representation

