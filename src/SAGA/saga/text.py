import os
import string
from collections.abc import Sequence

from loguru import logger

DEBUG = os.environ.get("DEBUG", "0") == "1"

STOP_WORDS = {
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "you're",
    "you've",
    "you'll",
    "you'd",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "she's",
    "her",
    "hers",
    "herself",
    "it",
    "it's",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "that'll",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "don't",
    "should",
    "should've",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "ve",
    "y",
    "ain",
    "aren",
    "aren't",
    "couldn",
    "couldn't",
    "didn",
    "didn't",
    "doesn",
    "doesn't",
    "hadn",
    "hadn't",
    "hasn",
    "hasn't",
    "haven",
    "haven't",
    "isn",
    "isn't",
    "ma",
    "mightn",
    "mightn't",
    "mustn",
    "mustn't",
    "needn",
    "needn't",
    "shan",
    "shan't",
    "shouldn",
    "shouldn't",
    "wasn",
    "wasn't",
    "weren",
    "weren't",
    "won",
    "won't",
    "wouldn",
    "wouldn't",
}


def clean_token(token: str) -> str:
    """Lowercase, strip, and drop punctuation to normalize tokens."""
    return token.strip().lower().translate(str.maketrans("", "", string.punctuation))


def is_content_word(token: str) -> bool:
    """Return True if token is non-empty after cleaning and not a stop word."""
    cleaned = clean_token(token)
    return bool(cleaned) and cleaned not in STOP_WORDS


def filter_generated_tokens(tokenizer, token_ids: Sequence[int], use_content_words: bool) -> list[int]:
    """
    Return indices of tokens to use. If use_content_words is True, keep only content
    words; otherwise return all indices. Falls back to all tokens if filtering empties.
    """
    if not use_content_words:
        return list(range(len(token_ids)))

    logger.debug(f"[Use content words: {use_content_words}] Only keeping content words")
    valid_indices = []
    for idx, token_id in enumerate(token_ids):
        token_str = tokenizer.decode([token_id], skip_special_tokens=False)
        if is_content_word(token_str):
            valid_indices.append(idx)
        elif DEBUG:
            logger.debug(f"Filtering out token '{token_str}' at index {idx}")

    if not valid_indices:
        return list(range(len(token_ids)))

    return valid_indices
