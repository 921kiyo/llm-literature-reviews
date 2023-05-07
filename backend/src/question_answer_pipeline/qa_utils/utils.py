import math
import string
from tqdm import tqdm


def maybe_is_text(s, thresh=2.5):
    if len(s) == 0:
        return False
    # Calculate the entropy of the string
    entropy = 0
    for c in string.printable:
        p = s.count(c) / len(s)
        if p > 0:
            entropy += -p * math.log2(p)

    # Check if the entropy is within a reasonable range for text
    if entropy > thresh:
        return True
    return False


def maybe_is_code(s):
    if len(s) == 0:
        return False
    # Check if the string contains a lot of non-ascii characters
    if len([c for c in s if ord(c) > 128]) / len(s) > 0.1:
        return True
    return False


def strings_similarity(s1, s2):
    if len(s1) == 0 or len(s2) == 0:
        return 0
    # break the strings into words
    s1 = set(s1.split())
    s2 = set(s2.split())
    # return the similarity ratio
    return len(s1.intersection(s2)) / len(s1.union(s2))


def maybe_is_truncated(s):
    punct = [".", "!", "?", '"']
    if s[-1] in punct:
        return False
    return True


def maybe_is_html(s):
    if len(s) == 0:
        return False
    # check for html tags
    if "<body" in s or "<html" in s or "<div" in s:
        return True


from .readers import parse_pdf
import tiktoken


def get_input_tokens(list_of_filenames, model='text-embedding-ada-002'):
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = 0

    docs_processed = {}

    for doc in tqdm(list_of_filenames):
        try:
            texts, _ = parse_pdf(doc, 'None', 'None', chunk_chars=3000)
            num_tokens = 0
            for text in texts:
                encoded_text = encoding.encode(text)
                num_tokens += len(encoded_text)

            docs_processed[doc] = num_tokens
        except:
            docs_processed[doc] = False

    total_tokens = sum(docs_processed.values())

    return total_tokens, docs_processed



