#!/usr/bin/env python3
"""
A script that contains the function bag_of_words()
that creates a bag of words embedding matrix:
"""

from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    A function that creates a bag of words embedding matrix:
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    E = vectorizer.fit_transform(sentences)
    F = vectorizer.get_feature_names_out()
    return E.toarray(), F
