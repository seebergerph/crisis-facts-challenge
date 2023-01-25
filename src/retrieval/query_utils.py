import string
import nltk
import nltk.corpus as corpus
import nltk.stem.porter as porter


nltk.download("stopwords")
PUNCTUATION = string.punctuation
STEMMER = porter.PorterStemmer()
STOPWORDS = set(corpus.stopwords.words("english"))


def drop_punctuation(sentence):
    cleaned = sentence.translate(
        str.maketrans(
            "", "",
            string.punctuation
        )
    )
    return cleaned


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


def preprocess(sentence, stopwords, stem):
    sentence = drop_punctuation(sentence)

    tokens = [
        token.lower()
        for token in sentence.split()
    ]

    tokens = drop_stop_words(tokens) if stopwords else tokens
    tokens = stemming(tokens) if stem else tokens
    return tokens


def merge_terms(terms1, terms2, duplicates=False,
                stopwords=True, stem=False):
    if stopwords or stem:
        terms1 = preprocess(terms1, stopwords, stem)
        terms2 = preprocess(terms2, stopwords, stem)
    else:
        terms1 = [term.lower() for term in terms1.split()]
        terms2 = [term.lower() for term in terms2.split()]

    terms = terms1 + terms2
    if duplicates:
        return " ".join(terms)

    terms = list(set(terms))
    return " ".join(terms)