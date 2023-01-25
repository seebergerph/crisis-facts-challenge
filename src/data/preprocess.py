from enum import Enum

import re
import string
import nltk
import nltk.corpus as corpus
import nltk.stem.porter as porter
import preprocessor as p
import wordsegment as ws


nltk.download("stopwords")
PUNCTUATION = string.punctuation
STEMMER = porter.PorterStemmer()
STOPWORDS = set(corpus.stopwords.words("english"))


def drop_stop_words(tokens):
    filtered_tokens = [
        token
        for token in tokens
        if token not in STOPWORDS
    ]
    return filtered_tokens


def stemming(tokens):
    stemmed_tokens = [
        STEMMER.stem(token)
        for token in tokens
    ]
    return stemmed_tokens


def drop_punctuation(sentence):
    cleaned = sentence.translate(
        str.maketrans(
            "", "",
            string.punctuation
        )
    )
    return cleaned


class Source(Enum):
    Twitter = "twitter"
    News = "news"
    Reddit = "reddit"
    Facebook = "facebook"


class DocumentPreprocessor:
    def __init__(self):
        p.set_options(
            p.OPT.URL, p.OPT.EMOJI,
            p.OPT.MENTION, p.OPT.RESERVED,
            p.OPT.SMILEY, p.OPT.ESCAPE_CHAR
        )
        ws.load()
        self.hashtag_pattern = re.compile(r"#\w*")
        self.whitespace_pattern = re.compile(r"\s{2,}")


    def _tweet(self, doc:str):
        doc = p.clean(doc)
        for hashtag in self.hashtag_pattern.findall(doc):
            words = " ".join(ws.segment(hashtag))
            doc = doc.replace(hashtag, words)
        doc = re.sub(self.whitespace_pattern, " ", doc)
        return doc


    def _news(self, doc:str):
        # Implement if required
        return doc


    def _reddit(self, doc:str):
        # Implement if required
        return doc


    def _facebook(self, doc:str):
        # Implement if required
        return doc


    def preprocess(self, doc:str, source:str):
        if source.lower() == Source.Twitter.value:
            return self._tweet(doc)
        if source.lower() == Source.News.value:
            return self._news(doc)
        if source.lower() == Source.Reddit.value:
            return self._reddit(doc)
        if source.lower() == Source.Facebook.value:
            return self._facebook(doc)
        return doc