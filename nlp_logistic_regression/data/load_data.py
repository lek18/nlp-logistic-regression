import csv
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(fpath: str) -> tuple[List[List[float]], List[int], TfidfVectorizer]:
    # map ham -> 0, spam -> 1
    cat_map = {"ham": 0, "spam": 1}
    tfidf = TfidfVectorizer()
    msgs, y = [], []
    filein = open(fpath, "r")
    reader = csv.reader(filein)
    for i, line in enumerate(reader):
        if i == 0:
            # skip over the header
            continue
        cat, msg = line
        y.append(cat_map[cat])
        msg = msg.strip()  # remove newlines
        msgs.append(msg)
    X = tfidf.fit_transform(msgs)
    return X, y, tfidf


def featurize(text, tfidf):
    features = tfidf.transform(text)
    return features
