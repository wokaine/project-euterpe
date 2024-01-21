import nltk
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re, sys

PUNC = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

class Normalizer():
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    def tokenize_lyrics(lyrics):
        stops = set(stopwords.words('english'))

        # Adding spaces between lines for Genius lyrics
        pattern = re.compile(r'([a-z])([A-Z])')
        lyrics = pattern.sub(r'\1 \2', lyrics)

        # Regex tokenizes words and completely tokenizes anything in brackets so they can be removed
        tokenizer = RegexpTokenizer(r'\[[^\[\]]*\]|\b\w+\b')
        tokens = tokenizer.tokenize(lyrics)

        pplyrics = [t.lower() for t in tokens if t.lower() not in stops and '(' not in t and '[' not in t]
        if '[' in pplyrics or ']' in pplyrics or '(' in pplyrics or ')' in pplyrics or len(pplyrics) == 0:
            print(pplyrics)
            sys.exit("Tokenization error")
        return set(pplyrics)

    def full_preprocess(lyrics):
        tokens = Normalizer.tokenize_lyrics(lyrics)
        lemmatizer = WordNetLemmatizer()
        stemmer = PorterStemmer()
        lemmatized_tokens = [lemmatizer.lemmatize(w, Normalizer.get_wordnet_pos(w)) for w in tokens]
        stemmed_tokens = [stemmer.stem(w) for w in lemmatized_tokens]

        return stemmed_tokens


    def create_query(string_songname, string_artistname):
        # Genius lyrics creates a query with:
        # The artists name, first letter capitalised, punctuation removed e.g. Anderson .Paak -> Anderson-paak
        # The song name, lowercase, no punctuation
        # Dashes instead of spaces

        processed_songname = string_songname.lower()
        processed_songname = "".join([c for c in processed_songname if c not in PUNC])
        processed_songname = processed_songname.replace(' ','-')

        processed_artistname = string_artistname.lower()
        processed_artistname = [c for c in processed_artistname if c not in PUNC]
        processed_artistname[0] = processed_artistname[0].upper()
        processed_artistname = "".join(processed_artistname)
        processed_artistname = processed_artistname.replace(' ','-')

        query = 'https://genius.com/'+processed_artistname+'-'+processed_songname+'-'+'lyrics'
        print(query)

        return query