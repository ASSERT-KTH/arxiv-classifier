# ML based Arxiv RSS feeds

This project aims at providing a feed of Arxiv papers on a given topic, not relying on the category metadata, but based on a classifier. 

For example, it's primary usage is to find those software engineering papers in arXiv that do not include the cs.SE tag.

## Preparing

Install dependencies with `pip install -r requirements.txt`.

To train a new model, download the arxiv dataset from Kaggle: <https://www.kaggle.com/Cornell-University/arxiv> and extract `arxiv-metadata-oai-snapshot.json` in the current directory.

## Running

### Training

Run [train.py](train.py) to train a new model with most recent data.
This does the following:

1. Train a model on a random half of the dataset. Uses binary classification where 1=cs.SE paper and 0=other paper. False positives are potentially mislabeled papers in the arxiv dataset and they are saved in file `false-positives0.txt`.
2. Train a model on the other half. False positives are saved in `false-positives1.txt`.
3. Train a model on all data. This model can be used to classify new papers with the fetch script.

### Fetching

Run [fetch.py](fetch.py) to fetch today's papers from arXiv's OAI-PMH interface.
Papers that are classified as software engineering but do not include `cs.SE` OR `cs.PL` tags are exported in an RSS feed.
