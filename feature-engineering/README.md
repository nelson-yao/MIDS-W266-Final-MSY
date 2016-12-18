### Feature Engineering

#### One Hot Encoding

The items categorizing the type of information discussed within the document were extracted, cleaned, clustered (manually) and converted into a [One Hot Encoding](https://en.wikipedia.org/wiki/One-hot) feature vector.

#### Word Embeddings
We used Stanford's geometric embedding methodology [GloVe](http://nlp.stanford.edu/projects/glove/) (which mainly consists of a Skipgram model with Singular Value Decomposition). The following implementations were used:
* Pre-trained GloVe 100d and 300d based on the Common Crawl (42B uncased tokens, 1.9M vocab)
* Self-trained GloVe word embedding from a vocabulary from the training set.

_The word embeddings implementation is in the `wordgeom/` subfolder._

#### Input Features
The main features were extracted from the pure text. For this several symbol and stopword removal was used. We experimented by changing the size of the stopword list and obtained better results as we increased the set of stopwords, however no sensitivity testing was performed. Future steps could include generating a TF-IDF / text-PCA to obtain the key terms from the documents.

For the final feature set we used two different approaches:
* A Bag of Words approach in which the count of all (non-stopword) terms using scikit-learn’s CountVectorizer implementation. This approach assumes that the order of the words is not relevant for the classification task at hand. The rationale behind this could be that the vocabulary of this documents is more technical and there is a differentiation between negative and positive terms, with limited use of negation modifiers.
* Ordered stream of words (with an additional stopword removal). In order, to capture the order of words we also implemented a stream of words feature set.
