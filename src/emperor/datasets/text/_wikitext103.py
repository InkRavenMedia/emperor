import re

from datasets import load_dataset


def load_raw_wikitext103(split, root):
    return load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split=split,
        cache_dir=str(root),
    )


def is_article_heading(text: str) -> bool:
    # A level-one heading starts a document. Subsections use repeated equals.
    return re.fullmatch(r"=\s+[^=\s].*?\s+=", text.strip()) is not None
