#Dictionary with contractions as keys and their expanded forms as values. 

CONTRACTION_MAP = CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

#Imports pandas for dataframe, re for regex in removing special characters, Beautiful soup for parsing html, ToktokTokenizer as a tokenizer, unicodedate to be used in removing special characters, contractions for contraction mapping, spacy for lemmatization and preprocessing, nltk for preprocessing 

import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.tokenize.toktok import ToktokTokenizer
import unicodedata
import contractions
import spacy
import nltk
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt_tab", quiet=True)

#Creating class Text Normalizer which takes a corpus and preprocesses it for NLP
#self.tokenizer - ToktokTokenizer as tokenizer
#self.stopword - list of stopwords to be removed
#self.nlp - loads Spacy English, small version 

class TextNormalizer:
    def __init__(self):
        self.tokenizer = ToktokTokenizer()
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
	
#strip_html_tags - strips html characters from document using beautiful soup and regex. #Returns stripped text

    def strip_html_tags(self, text):
        soup = BeautifulSoup(text, "html.parser")
        [s.extract() for s in soup(['iframe', 'script'])]
        stripped_text = soup.get_text()
        stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
        return stripped_text

#remove_accented_chars - removes accented characters from text

    def remove_accented_chars(self, text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

#expand_contractions - Expands contractions. References CONTRACTION_MAP

    def expand_contractions(self, text, contraction_mapping):
        contractions_pattern = re.compile(
            '({})'.format('|'.join(contraction_mapping.keys())),
            flags=re.IGNORECASE | re.DOTALL
        )

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match)
            if not expanded_contraction:
                expanded_contraction = contraction_mapping.get(match.lower())
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction

        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text

#lemmatize_text - uses SpaCy to reduce words to their lemmas. 

    def lemmatize_text(self, text):
        doc = self.nlp(text)
        return ' '.join([token.lemma_ if token.lemma_ != '-PRON-' else token.text for token in doc])

    def remove_special_characters(self, text, remove_digits=False):
        pattern = r'[^a-zA-z\s]'
        text = re.sub(pattern, '', text)
        return text

#remove_stopwords - removes any stopwords listed in stopword_list 


#normalize_series - This is for normalizing a corpus in form of line series. takes self, #corpus, contraction_mapping, and all of the above function as parameters. (Above #functions are Boolean parameters.)Applies functions to corpus (unless False)

    def normalize_series(self, corpus, contraction_mapping=CONTRACTION_MAP,
                         html_stripping=True,
                         contraction_expansion=True,
                         accented_char_removal=True,
                         text_lower_case=True,
                         text_lemmatization=True,
                         special_char_removal=True,
                         stopword_removal=True,
                         remove_digits=True):
        normalized_corpus = corpus.copy()

        if html_stripping:
            normalized_corpus = normalized_corpus.apply(lambda x: self.strip_html_tags(x))

        if contraction_expansion:
            normalized_corpus = normalized_corpus.apply(lambda x: self.expand_contractions(x, contraction_mapping))

        if text_lower_case:
            normalized_corpus = normalized_corpus.apply(lambda x: x.lower())

        normalized_corpus = normalized_corpus.apply(lambda x: re.sub(r'[\r\n]+', ' ', x))

        if text_lemmatization:
            normalized_corpus = normalized_corpus.apply(lambda x: self.lemmatize_text(x))

        if special_char_removal:
            normalized_corpus = normalized_corpus.apply(lambda x: self.remove_special_characters(x))

        normalized_corpus = normalized_corpus.apply(lambda x: re.sub(r'\s+', ' ', x).strip())


#returns normalized corpus 
        
        return normalized_corpus
#For normalizing corpus that's a string, or something to that effect. 

    def normalize_corpus(self, corpus, contraction_mapping=CONTRACTION_MAP,
                         html_stripping=True,
                         contraction_expansion=True,
                         accented_char_removal=True,
                         text_lower_case=True,
                         text_lemmatization=True,
                         special_char_removal=True,
                         remove_digits=True):

