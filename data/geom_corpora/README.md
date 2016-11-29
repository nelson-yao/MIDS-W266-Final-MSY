## Geometric Embedding Corpora

This folder contains the saved geometric embeddings copora.

__Note:__
_Each lookup-corpus is not a collection of documents (in the conventional sense of corpus), but rather a dictionary with_ `{ term: representation_vector }`

Only gloVe lookup-corpora are contained here.

The default gloVe lookup-corpora are available for download from the following links:
 *  [gloVe trained on](https://s3.amazonaws.com/nlp-mids-266-project/geom_corpora/glove.6B.100d.txt.zip): Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, **100d vectors**, 128MB download)
 *  [gloVe trained on](https://s3.amazonaws.com/nlp-mids-266-project/geom_corpora/glove.6B.300d.txt.zip): Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, **300d vectors**, 376MB download)

_Files need to be unzipped for the representation to be read._

The original *zipped* corpora may be downloaded from [stanfordnlp](http://nlp.stanford.edu/data/wordvecs/)
