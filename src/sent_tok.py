from flair.embeddings import WordEmbeddings

from flair.embeddings import FlairEmbeddings
from flair.embeddings import WordEmbeddings, FlairEmbeddings
from flair.embeddings import StackedEmbeddings
from flair.models import SequenceTagger
from flair.models import MultiTagger
from flair.tokenization import SegtokSentenceSplitter
from flair.data import Sentence

from flair.embeddings import FlairEmbeddings, TransformerWordEmbeddings
from flair.embeddings import StackedEmbeddings

# init Flair embeddings
flair_forward_embedding = FlairEmbeddings('multi-forward')
flair_backward_embedding = FlairEmbeddings('multi-backward')

# init multilingual BERT
bert_embedding = TransformerWordEmbeddings('bert-base-multilingual-cased')


# now create the StackedEmbedding object that combines all embeddings
stacked_embeddings = StackedEmbeddings(
    embeddings=[flair_forward_embedding, flair_backward_embedding, bert_embedding])

# load tagger
tagger = MultiTagger.load(['de-pos', 'pos'])
splitter = SegtokSentenceSplitter()

def sent_tok (text):
    new_text_array = []
    sentences = splitter.split(text)
    
    tags = tagger.predict(sentences)

    for sent in sentences:
        embeddings = stacked_embeddings.embed(sent)
        for token in sent:
            new_text_array.append(token.embedding)

        #print(sent.to_tagged_string())
        #print(sent)
    return new_text_array