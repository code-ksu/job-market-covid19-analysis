from multiprocessing import Pool, TimeoutError
import time
import os
import string

from HanTa import HanoverTagger as ht
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import sent_tokenize

tagger = ht.HanoverTagger('morphmodel_ger.pgz')
nltk.download('stopwords')
nltk.download('punkt')
stopwords_de = stopwords.words('german')
stopwords_en = stopwords.words('english')
stopwords_de_extra = ['Schwerpunkt', 'Bereich', 'Mitarbeiter', 'Sachbearbeiter', 'Professional', 'sowie']

def lemma_text(text):
    new_text_array = []
    #new_text = ''
    sentences = sent_tokenize(text, language = 'german')
    for sent in sentences:
        tokens = word_tokenize(sent, language = 'german')
        lemma = [lemma.lower() for (word, lemma, pos) in tagger.tag_sent(tokens)]
        words = [w for w in lemma if not w in stopwords_de]
        words = [w for w in words if not w in stopwords_de_extra]
        words = [e for e in words if not e in stopwords_en]
        words = [word for word in words if word.isalpha()]
        #new_text += ' '.join(words) + ' ' #for ngramms we don't need this line of code
        new_text_array.extend(words)
        #return new_text.strip()
    return new_text_array
