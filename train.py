import json
import logging
import os
import pickle
import random

import pandas as pd
from scipy.special import softmax
from simpletransformers.classification import ClassificationModel

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

MODEL = ("roberta", "roberta-base")
# MODEL = ("mobilebert", "google/mobilebert-uncased")
# MODEL = ("distilbert", "distilbert-base-uncased")


def single_line(s):
    return " ".join(s.split())


def init_data(limit=0):
    # Download from https://www.kaggle.com/Cornell-University/arxiv
    with open("arxiv-metadata-oai-snapshot.json") as f:
        for idx, line in enumerate(f):
            entry = json.loads(line)

            cs = se = False
            for cat in entry["categories"].lower().split():
                if cat in ("cs.se", "cs.pl"):
                    cs = se = True
                    break
                if cat.startswith("cs.") or cat == "stat.ml":
                    cs = True
            if not cs:
                continue

            yield [
                entry["id"],
                single_line(entry["title"])
                + " abstract: "
                + single_line(entry["abstract"]),
                int(se),
            ]
            if limit > 0 and idx >= limit:
                break


def repeat_1(data):
    """
    Simple oversampling for the '1' class due to heavy imbalancing
    """
    for x in data:
        label = x[-1]
        if label == 1:
            for _ in range(10):
                yield x
        else:
            assert label == 0
            yield x


data = list(init_data())
random.shuffle(data)
split_idx = int(len(data) * 0.5)
for idx in range(3):
    if idx == 0:
        train_data = data[:split_idx]
        test_data = data[split_idx:]
    elif idx == 1:
        train_data = data[split_idx:]
        test_data = data[:split_idx]
    else:
        train_data = data
        test_data = None
    train_data = list(repeat_1(train_data))
    random.shuffle(train_data)

    train_data = pd.DataFrame(data=train_data, columns=["arxivId", "text", "isSE"])
    train_data.to_csv(f"train{idx}.csv")
    if test_data is not None:
        test_data = pd.DataFrame(data=test_data, columns=["arxivId", "text", "isSE"])
        test_data.to_csv(f"test{idx}.csv")

    model = ClassificationModel(
        *MODEL,
        args={
            "train_batch_size": 64,
            "eval_batch_size": 64,
            "process_count": 8,
            "save_eval_checkpoints": False,
            "output_dir": f"outputs{idx}/",
        },
    )  # , num_labels=2, use_cuda=True)
    model.train_model(train_data.drop("arxivId", 1))

    if test_data is None:
        continue

    result, model_outputs, wrong_predictions = model.eval_model(
        test_data.drop("arxivId", 1)
    )
    print(idx, result)
    with open(f"result{idx}.pickle", "wb") as f:
        pickle.dump([result, model_outputs, wrong_predictions], f)

    # TODO: include year
    # XXX: Sort can include age. E.g. (prob is SE/PL) - (age in years) / 5
    with open(f"false-positives{idx}.txt", "w") as f:
        for p, x in sorted(
            ((model_outputs[x.guid], x) for x in wrong_predictions if x.label == 0),
            key=lambda x: x[0][0],
        ):
            print(
                r"https://arxiv.org/abs/" + test_data["arxivId"][x.guid],
                softmax(p)[1],
                x.text_a,
                file=f,
            )
