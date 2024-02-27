import functools
import logging

from bs4 import BeautifulSoup
from langchain_community.document_loaders.recursive_url_loader import \
    RecursiveUrlLoader
from langchain_core.load import dumps, loads
from langgraph_engineer.constants import DOCS_PATH
import warnings

logger = logging.getLogger(__name__)


def html_to_markdown(tag):
    if tag.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        level = int(tag.name[1])
        return f"{'#' * level} {tag.get_text()}\n\n"
    elif tag.name == "pre":
        code_content = tag.find("code")
        if code_content:
            return f"```\n{code_content.get_text()}\n```\n\n"
        else:
            return f"```\n{tag.get_text()}\n```\n\n"
    elif tag.name == "p":
        return f"{tag.get_text()}\n\n"
    return ""


def clean_document(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    markdown_content = ""
    for child in soup.recursiveChildGenerator():
        if child.name:
            markdown_content += html_to_markdown(child)
    return markdown_content


def ingest(dry_run: bool = False):
    logger.info("Ingesting documents...")
    # LangGraph docs
    url = "https://python.langchain.com/docs/langgraph/"
    loader = RecursiveUrlLoader(
        url=url, max_depth=20, extractor=lambda x: clean_document(x)
    )
    docs = loader.load()

    # Sort the list based on the URLs in 'metadata' -> 'source'
    d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
    d_reversed = list(reversed(d_sorted))

    if dry_run:
        print(_format_docs(d_reversed))
        return
    # Dump the documents to 'DOCS_PATH'
    docs_str = dumps(d_reversed)
    with DOCS_PATH.open("w") as f:
        f.write(docs_str)
    logger.info("Documents ingested.")


def _format_docs(docs):
    return "\n\n\n --- \n\n\n".join([doc.page_content for doc in docs])


@functools.lru_cache
def load_docs() -> str:
    # Load the documents from 'DOCS_PATH'
    if not DOCS_PATH.exists():
        logger.warning("No documents found. Ingesting documents...")
        ingest()
    with DOCS_PATH.open("r") as f:
        # Suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d_reversed = loads(f.read())

    # Concatenate the 'page_content'
    return _format_docs(d_reversed)
