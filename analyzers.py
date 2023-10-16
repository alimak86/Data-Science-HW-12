import spacy

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

from heapq import nlargest


class spacy_analyzer:
    def __init__(self,text):
        self.nlp = spacy.load('en_core_web_sm')
        self.doc = self.nlp(text)
        self.tokens = [token.text for token in self.doc]
        self.sentences = list(self.doc.sents)

    def summary(self,num_max_elem):
        sentence_scores = {sent: len(sent) for sent in self.sentences}
        summary_sentences = nlargest(num_max_elem, sentence_scores, key=sentence_scores.get)
        summary = ' '.join([sent.text for sent in summary_sentences])
        return summary


class nltk_analyser:
    punctuation=""
    def __init__(self,text):
        nltk.download('punkt')
        nltk.download('stopwords')
        self.tokens = word_tokenize(text)
        self.sentences = sent_tokenize(text)
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = self.punctuation + '\n'
        self.sentence_scores = {}
        for sentence in self.sentences:
            word_count = len([word for word in sentence.split() if word.lower() not in self.stop_words])
            self.sentence_scores[sentence] = word_count
    
    def summary(self,num_max_elem):
        summary_sentences = nlargest(num_max_elem, self.sentence_scores, key=self.sentence_scores.get)
        summary = ' '.join(summary_sentences)
        return summary