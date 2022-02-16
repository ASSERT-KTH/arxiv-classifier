#!/usr/bin/env python3
import logging
import os
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
import glob
import sys
import feedgenerator
import requests
from scipy.special import softmax

logging.basicConfig()
log = logging.getLogger(__name__)


def main():
    harvest_since_last_modification()
    classify()

def classify(category):
    model = load_model()
    i=0
    for entry in iter_load_entries_from_xml():
        output_file = entry['file']+".class.txt"
        # TODO: batch predictions, it will go much faster
        if not os.path.exists(output_file):
            i+=1
            # bounding the number of predictions
            if i>10000: 
                print("max number of classifications reached")
                return
            texts = [
                single_line(entry["title"] + " abstract: " + entry["abstract"]) ]        
            pred, score = model.predict(texts) 
            
            title = entry["title"].replace('\n',' ')
            with open(output_file,"w") as f: f.write(str(pred[0])+" "+title)
            print(i,"\r", end="")
            if pred[0] == 1 and  category not in entry["categories"].lower():
                # this is the most simple console output
                print(i, title,entry["link"])
            #print(pred, score)
        else:
            # already predicted
            pass

def generate_feed(category):
    model = load_model()
    entries = list(iter_load_entries_from_xml())
    texts = [
        single_line(entry["title"] + " abstract: " + entry["abstract"])
        for entry in entries
    ]

    entries = (
        *model.predict(texts),
        texts,
        entries,
    )  # prediction label, score, arxiv text, arxiv label

    feed = feedgenerator.Rss201rev2Feed(
        title="arXiv misclassified: all",
        link="http://export.arxiv.org/rss/",
        description="Papers from arXiv that look like "+category+".",
        language="en",
    )

    for pred, score, text, entry in zip(*entries):
        label = entry["categories"]
        if pred and "cs.se" not in label.lower() and "cs.pl" not in label.lower():
            abs_link = entry["link"]
            abstract = entry["abstract"]
            authors = entry["authors"]
            pdf_link = abs_link.replace("/abs/", "/pdf/")
            score = softmax(score)
            title = entry["title"]

            r = requests.get(abs_link)
            if r.ok:
                description = r.text
            else:
                description = f"""
                {abstract}
                <p>Authors: {authors}
                <p><a href="{pdf_link}">{pdf_link}</a>
                <p><a href="{abs_link}">{abs_link}</a>
                <p>Categories: {label}
                <p>score: {score[1]:.2f}
                """.strip()

            args = dict(
                title=title,
                link=pdf_link,
                description=description,
                unique_id=pdf_link,
                categories=label.split(),
            )
            feed.add_item(**args)


    os.makedirs("feed", exist_ok=True)
    with open("feed/feed.xml", "w") as f:
        print(feed.writeString("utf-8"), file=f)



def harvest_since_last_modification(date = datetime.today()):
    formatted_date = date.strftime("%Y-%m-%d")
    log.info("Harvesting since %s", date)
    subprocess.run(
        f"mkdir -p data && oai-harvest 'http://export.arxiv.org/oai2' --dir data --from {formatted_date} -p arXiv",
        check=True,
        shell=True,
    )


def iter_load_entries_from_xml():
    tags = ("abstract", "authors", "categories", "id", "title")

    for fname in glob.iglob("data/*.xml"):
        root = ET.parse(fname).getroot()
        d = {"file":fname}
        for el in root:
            tag = el.tag
            for wanted_tag in tags:
                if tag.endswith(wanted_tag):
                    d[wanted_tag] = el_text(el)

        if all(tag in d for tag in tags):  # Sanity check: valid entry
            d["link"] = f"https://arxiv.org/abs/{d['id']}"
            yield d
        else:
            log.warning(
                "File %s is not complete, contains keys: %s", fname, list(d.keys())
            )


def el_text(el):
    if not el.tag.endswith("authors"):
        return el.text.strip()
    return " - ".join(author_names_text(el))


def author_names_text(el):
    for child in el:
        yield " ".join(child.itertext())


def single_line(s):
    return " ".join(s.split())


def load_model():
    from simpletransformers.classification import ClassificationModel

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    log.debug("Loading ClassificationModel")
    return ClassificationModel(
        "roberta",
        "outputs/",
        use_cuda=False,
        args={"train_batch_size": 64, "eval_batch_size": 64, "process_count": 8, "silent":True},
    )


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        harvest_since_last_modification(datetime.utcfromtimestamp(max([os.path.getmtime(x) for x in glob.iglob("data/*")])))
    if sys.argv[1] == "classify":
        classify(sys.argv[2])
    if sys.argv[1] == "feed":
        generate_feed(sys.argv[2])
