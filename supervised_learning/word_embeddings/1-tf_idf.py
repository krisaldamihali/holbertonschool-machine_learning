#!/usr/bin/env python3
"""
A scipt that contains the function tf_idf()
that creates a TF-IDF embedding matrix:
"""

import scipy.cluster.hierarchy
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    A function that creates a TF-IDF embedding matrix:
    """
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    E = vectorizer.fit_transform(sentences)
    F = vectorizer.get_feature_names_out()
    return E.toarray(), F
