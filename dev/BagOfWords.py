"""Holy kek, this is a bag of words implementation."""
import numpy as np
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop = set(stopwords.words("english"))
# include punctuation in stop words
stop.update(['.', ',', '"', "'", '’', '‘', '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])

from sklearn.feature_extraction.text import CountVectorizer


class BoW():
    """Initializes a bag of words object for a book. 
    Usage (see bag_of_words.ipynb for more example):
        bow = BoW(book)
        bow.look_up("The Origins of the Second World War", top_hits=5)  # returns the top 5 indices of the book_string
    """
    vectorizer = CountVectorizer()
    cache = {}
    def __init__(self, book_title, embedding_length=None):
        """Provided a book list(string), this class will create a bag of words or corpus for the book."""
        if embedding_length is None:  # book_title is embeddings as saved in result_generation.iåpynb
            context_embeddings = book_title        
        else:
            with open(f'Neural Search/Embeddings/{book_title}_context_embeddings_{embedding_length}.pkl', 'rb') as f:
                context_embeddings = pickle.loads(f.read())

        book = list(context_embeddings.keys())  # keys are natural language strings

        self.book = book
        self.corpus = self._convert(book)
    
    def _convert(self, book):
        filtered_book = [" ".join([word.lower() for word in word_tokenize(context) if word.lower() not in stop]) for context in book]
        return np.array(self.vectorizer.fit_transform(filtered_book).toarray())
    
    def look_up(self, query, top_hits=1):
        """Returns the >top_hits< indices of similar contexts in the corpus"""
        assert top_hits <= len(self.corpus), "top_hits must be less than or equal to the number of documents in the corpus"
        
        # filter query and convert to BoW
        query = " ".join([word.lower() for word in word_tokenize(query) if word.lower() not in stop])
        query = self.vectorizer.transform([query]).toarray()

        # find top hits as similarity and return indices
        product = np.array(query[0]) @ self.corpus.T
        return product.argsort()[-top_hits:][::-1]
    
    @staticmethod
    def compare(query, ctx):
        """Returns the similarity between query and ctx"""
        # First convert query and ctx to BoW and then find similarity with dot product
        vectorizer = CountVectorizer()  # new vectorizer to avoid changing the old one

        ctx = " ".join([word.lower() for word in word_tokenize(ctx) if word.lower() not in stop])
        ctx_bow = vectorizer.fit_transform([ctx]).toarray()

        query = " ".join([word.lower() for word in word_tokenize(query) if word.lower() not in stop])
        try:
            query_bow = vectorizer.transform([query]).toarray()
        except ValueError:
            print("ValueError: empty vocabulary; perhaps the documents only contain stop words")
            return 0

        return ctx_bow[0] @ query_bow[0]
        # Using cache (not working)
        # Look up query and ctx in cache
        # if ctx in self.cache:
        #     ctx_bog = self.cache[ctx]
        # else:
        #     ctx = " ".join([word.lower() for word in word_tokenize(ctx) if word.lower() not in stop])
        #     ctx_bog = vectorizer.fit_transform([ctx]).toarray()
        #     self.cache[ctx] = ctx_bog
        
        # if query in self.cache:
        #     query_bog = self.cache[query]
        #     if query_bog is None:
        #         return 0
        # else:
        #     query = " ".join([word.lower() for word in word_tokenize(query) if word.lower() not in stop])
        #     try:
        #         query_bog = vectorizer.transform([query]).toarray()
        #         self.cache[query] = query_bog
        #     except ValueError:
        #         print("ValueError: empty vocabulary; perhaps the documents only contain stop words")
        #         self.cache[query] = None
        #         return 0


