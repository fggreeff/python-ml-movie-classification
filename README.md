# python-ml-movie-classification

Python flask application which does prediction on movie reviews using nltk

## Getting started

These instructions will get you a copy of the project up and running on your local machine

```$ git clone git@github.com:fggreeff/python-ml-movie-classification.git```

## Prerequisites

- Python 3.7 (You could have multiple versions of python, use `which python python2 python3` for the version and corresponding location)

- pip (same as above `which pip pip3`)

## Local setup

- It’s best practice to keep all your virtualenvs in one place. Create a folder `mkdir ~/.virtualenvs`

- Setup a virtual env `python3 -m venv ~/.virtualenvs/movie-classifier`

- Active env `source ~/.virtualenvs/movie-classifier/bin/activate`

- Install dependencies `pip3 install -r requirements.txt`

## Running locally

- The movie_reviews data set is used and can be downloaded using the following command from a terminal:
`python -m nltk.downloader movie_reviews`. 
More information on [nltk howto corpus](http://www.nltk.org/howto/corpus.html) and [nltk corpus tutorial](https://pythonprogramming.net/nltk-corpus-corpora-tutorial/)

- Download and train prediction model `python3 classifier_train.py`

- Run prediction service `python3 classifier_predict.py`. 
View application status running [locally here](http://localhost:5555/)

- Make a POST request to [localhost:5555/predict](http://localhost:5555/predict) or import the postman collection.

Example JSON body request:
```{"document": "I like this movie"}```

## Postman collection

[![Run in Postman](https://run.pstmn.io/button.svg)](https://app.getpostman.com/run-collection/39fe36eeb99cf59b78ed) Endpoint listed in the collection

[Instructions on how to import into Postman](https://learning.getpostman.com/docs/postman/collections/data_formats/#importing-postman-data)

## Source

[Resource](https://www.linkedin.com/company/frameworktraining/)