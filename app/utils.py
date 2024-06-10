import logging
import pickle
from functools import wraps

from langchain_community.vectorstores import FAISS


def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info("Starting %s", func._name_)
        result = func(*args, **kwargs)
        logging.info("%s completed successfully", func._name_)
        return result

    return wrapper


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@log_execution
def load_knowledgebase(path, embeddings):
    if not path.endswith("pickle"):
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    with open(path, "rb") as f:
        result = pickle.load(f)
    return result
