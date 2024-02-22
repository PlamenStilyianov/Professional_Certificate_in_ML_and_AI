from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize

wnl = WordNetLemmatizer()
sent = 'This is a foo bar sentence'
pos_tag(word_tokenize(sent))

for word, tag in pos_tag(word_tokenize(sent)):
    wntag = tag[0].lower()
    wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
    if not wntag:
        lemma = word
    else:
        lemma = wnl.lemmatize(word, wntag)
    print(lemma)
