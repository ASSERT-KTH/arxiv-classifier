# Automatic categorization for Arxiv

This project aims at providing a feed of Arxiv papers on a given topic, not relying on the category metadata, but based on a classifier. 

The core idea is to train the system based on the existing categories, and subsequently use the classifier to predict papers that could have been tagged with a given category.

For example, we it to find those software engineering papers in arXiv that do not include the cs.SE tag.

## Preparing

Install dependencies with `pip install -r requirements.txt`.

Get a model:

* Download an already trained model and uncompress it. This creates a folder `outputs`.
  * Model for cs.SE at <https://github.com/kth-tcs/arxiv-classifier/releases/download/v0/model.tar.bzip2> 
* Train a new model. (share it if you do so!)


## Running

### Training

To train a new model, download the arxiv dataset from Kaggle: <https://www.kaggle.com/Cornell-University/arxiv> and extract `arxiv-metadata-oai-snapshot.json` in the current directory.

Run [train.py](train.py) to train a new model with most recent data.
This does the following:

1. Train a model on a random half of the dataset. Uses binary classification where 1=cs.SE paper and 0=other paper. False positives are potentially mislabeled papers in the arxiv dataset and they are saved in file `false-positives0.txt`.
2. Train a model on the other half. False positives are saved in `false-positives1.txt`.
3. Train a model on all data. This model can be used to classify new papers with the fetch script.

### Run


To fetch recent papers from arXiv's OAI-PMH interface: `./run.py fetch`
This creates XML file in folder `./data`.

To classify papers: `./run.py classify cs.se`.
This outputs the missing papers on the console and saves the classification in `./data/*.class.txt`.

To create an RSS feed: `./run.py feed cs.se`
