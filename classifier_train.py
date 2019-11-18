"""
This script implements the training step of a classification system.

After the training, it saves the classifier and vectoriser as
pickle files, to be used by the classifier_predict.py script.
"""
import pickle
from nltk.corpus import movie_reviews as data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


if __name__ == '__main__':
    
    # Get the data
    corpus = [data.raw(fileid) for fileid in data.fileids('pos')]
    corpus += [data.raw(fileid) for fileid in data.fileids('neg')]

    target = ['pos'] * 1000  # ['pos', 'pos', ... x1000]
    target += ['neg'] * 1000

    # Set-up the vectoriser
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.7)
    # Vectorise the data set
    vectorised_corpus = vectorizer.fit_transform(corpus)

    # Set-up the classifier
    classifier = LinearSVC()
    # Training
    classifier.fit(vectorised_corpus, target)

    # Save classifier and vectoriser as binary files,
    # to be re-used for prediction later
    with open('my_classifier.pickle', 'wb') as f:
        pickle.dump(classifier, f)
        print("Classifier saved")

    with open('my_vectoriser.pickle', 'wb') as f:
        pickle.dump(vectorizer, f)
        print("Vectoriser saved")
