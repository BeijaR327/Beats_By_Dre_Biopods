import re
from typing import Dict, List
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure resources (download at runtime if missing)
def _ensure_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

_ensure_nltk()

def _simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text)

def build_text_pipeline(cfg: Dict):
    lowercase = cfg.get('lowercase', True)
    strip_punct = cfg.get('strip_punct', True)
    remove_stop = cfg.get('remove_stopwords', True)
    lemmatize = cfg.get('lemmatize', True)

    stop = set(stopwords.words('english'))
    lemm = WordNetLemmatizer()

    class Pipe:
        def transform(self, texts):
            out = []
            for t in texts:
                if not isinstance(t, str):
                    t = '' if t is None else str(t)
                s = t
                if lowercase:
                    s = s.lower()
                toks = _simple_tokenize(s) if strip_punct else s.split()
                if remove_stop:
                    toks = [w for w in toks if w not in stop and len(w) > 1]
                if lemmatize:
                    toks = [lemm.lemmatize(w) for w in toks]
                out.append(' '.join(toks))
            return out
    return Pipe()
