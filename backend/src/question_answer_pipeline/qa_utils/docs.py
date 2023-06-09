from typing import List, Optional, Tuple, Dict, Callable, Any, Union
from functools import reduce
import os
import os
from pathlib import Path
import re
from .utils import maybe_is_text
from .qaprompts import (
    summary_prompt,
    qa_prompt,
    search_prompt,
    citation_prompt,
    make_chain,
)
from dataclasses import dataclass
from .readers import read_doc
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.llms.base import LLM
from langchain.callbacks import get_openai_callback
from langchain.cache import SQLiteCache
import langchain
from datetime import datetime

CACHE_PATH = Path.home() / ".paperqa" / "llm_cache.db"
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
langchain.llm_cache = SQLiteCache(CACHE_PATH)


@dataclass
class Answer:
    """A class to hold the answer to a question."""

    question: str
    answer: str = ""
    context: str = ""  # shows which documents were relevant to answer
    contexts: Dict[str, Tuple] = None  # dict(url= (key, citation, llm_summary, chunked_text))
    references: str = ""  # string for references
    formatted_answer: str = ""  # formatted answer to question with bibliography
    passages: Dict[str, str] = None
    tokens: int = 0
    question_embedding: List[float] = None
    from_embed: bool = False

    def __post_init__(self):
        """Initialize the answer."""
        if self.contexts is None:
            self.contexts = {}
        if self.passages is None:
            self.passages = {}

    def __str__(self) -> str:
        """Return the answer as a string."""
        return self.formatted_answer


class Docs:
    """A collection of documents to be used for answering questions."""

    def __init__(
            self,
            chunk_size_limit: int = 3000,
            llm: Optional[Union[LLM, str]] = None,
            summary_llm: Optional[Union[LLM, str]] = None,
            name: str = "default",
            index_path: Optional[Path] = None,
    ) -> None:
        """Initialize the collection of documents.



        Args:
            chunk_size_limit: The maximum number of characters to use for a single chunk of text.
            llm: The language model to use for answering questions. Default - OpenAI chat-gpt-turbo
            summary_llm: The language model to use for summarizing documents. If None, llm is used.
            name: The name of the collection.
            index_path: The path to the index file IF pickled. If None, defaults to using name in $HOME/.paperqa/name
        """
        self.docs = dict()  # self.docs[path] = dict(texts=texts, metadata=metadata, key=key)
        self.chunk_size_limit = chunk_size_limit
        self.keys = set()
        self._faiss_index = None
        self.update_llm(llm, summary_llm)
        if index_path is None:
            index_path = Path.home() / ".paperqa" / name
        self.index_path = index_path
        self.name = name

    def update_llm(
            self,
            llm: Optional[Union[LLM, str]] = None,
            summary_llm: Optional[Union[LLM, str]] = None,
    ) -> None:
        """Update the LLM for answering questions."""
        if llm is None:
            llm = "gpt-3.5-turbo"
        if type(llm) is str:
            llm = ChatOpenAI(temperature=0, model=llm)
        if type(summary_llm) is str:
            summary_llm = ChatOpenAI(temperature=0, model=summary_llm)
        self.llm = llm
        if summary_llm is None:
            summary_llm = llm
        self.summary_llm = summary_llm
        self.summary_chain = make_chain(prompt=summary_prompt, llm=summary_llm)
        self.qa_chain = make_chain(prompt=qa_prompt, llm=llm)
        self.search_chain = make_chain(prompt=search_prompt, llm=summary_llm)
        self.cite_chain = make_chain(prompt=citation_prompt, llm=summary_llm)

    def add(
            self,
            path: str,
            citation: Optional[str] = None,
            key: Optional[str] = None,
            disable_check: bool = False,
            chunk_chars: Optional[int] = 3000,
    ) -> None:
        """Add a document to the collection."""

        # first check to see if we already have this document
        # this way we don't make api call to create citation on file we already have
        if path in self.docs:
            raise ValueError(f"Document {path} already in collection.")

        if citation is None:
            # peak first chunk
            texts, _ = read_doc(path, "", "", chunk_chars=chunk_chars)
            with get_openai_callback() as cb:
                citation = self.cite_chain.run(texts[0])
            if len(citation) < 3 or "Unknown" in citation or "insufficient" in citation:
                citation = f"Unknown, {os.path.basename(path)}, {datetime.now().year}"

        if key is None:
            # get first name and year from citation
            try:
                author = re.search(r"([A-Z][a-z]+)", citation).group(1)
            except AttributeError:
                # panicking - no word??
                raise ValueError(
                    f"Could not parse key from citation {citation}. Consider just passing key explicitly - e.g. docs.py (path, citation, key='mykey')"
                )
            try:
                year = re.search(r"(\d{4})", citation).group(1)
            except AttributeError:
                year = ""
            key = f"{author}{year}"
        suffix = ""
        while key + suffix in self.keys:
            # move suffix to next letter
            if suffix == "":
                suffix = "a"
            else:
                suffix = chr(ord(suffix) + 1)
        key += suffix
        self.keys.add(key)

        texts, metadata = read_doc(path, citation, key, chunk_chars=chunk_chars)

        # loose check to see if document was loaded
        if len("".join(texts)) < 10 or (
                not disable_check and not maybe_is_text("".join(texts))
        ):
            raise ValueError(
                f"This does not look like a text document: {path}. Path disable_check to ignore this error."
            )

        self.docs[path] = dict(texts=texts, metadata=metadata, key=key)
        if self._faiss_index is not None:
            self._faiss_index.add_texts(texts, metadatas=metadata)

    def add_from_embeddings(
            self,
            path: str,
            texts,
            text_embeddings: List[float],
            metadatas

    ) -> None:

        """Add a document to the collection."""

        # first check to see if we already have this document
        # this way we don't make api call to create citation on file we already have
        if path in self.docs:
            raise ValueError(f"Document {path} already in collection.")

        key = metadatas[0]['dockey']

        suffix = ""
        while key + suffix in self.keys:
            # move suffix to next letter
            if suffix == "":
                suffix = "a"
            else:
                suffix = chr(ord(suffix) + 1)
        key += suffix
        self.keys.add(key)

        if key != metadatas[0]['dockey']:
            for j in range(len(metadatas)):
                metadatas[j]['dockey'] = key

        self.docs[path] = dict(texts=texts, metadata=metadatas, key=key)

        if self._faiss_index is not None:
            self._faiss_index.add_embeddings([*zip(texts, text_embeddings)],
                                             metadatas=metadatas)
        else:
            """Instantiate FAISS"""
            self._faiss_index = FAISS.from_embeddings([*zip(texts, text_embeddings)],
                                                      metadatas=metadatas,
                                                      embedding=OpenAIEmbeddings())

    def clear(self) -> None:
        """Clear the collection of documents."""
        self.docs = dict()
        self.keys = set()
        self._faiss_index = None
        # delete index file
        pkl = self.index_path / "index.pkl"
        if pkl.exists():
            pkl.unlink()
        fs = self.index_path / "index.faiss"
        if fs.exists():
            fs.unlink()

    @property
    def doc_previews(self) -> List[Tuple[int, str, str]]:
        """Return a list of tuples of (key, citation) for each document."""
        return [
            (
                len(doc["texts"]),
                doc["metadata"][0]["dockey"],
                doc["metadata"][0]["citation"],
            )
            for doc in self.docs.values()
        ]

    # to pickle, we have to save the index as a file
    def __getstate__(self):
        if self._faiss_index is None and len(self.docs) > 0:
            self._build_faiss_index()
        state = self.__dict__.copy()
        if self._faiss_index is not None:
            state["_faiss_index"].save_local(self.index_path)
        del state["_faiss_index"]
        # remove LLMs (they can have callbacks, which can't be pickled)
        del state["summary_chain"]
        del state["qa_chain"]
        del state["cite_chain"]
        del state["search_chain"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            self._faiss_index = FAISS.load_local(self.index_path, OpenAIEmbeddings())
        except:
            # they use some special exception type, but I don't want to import it
            self._faiss_index = None
        self.update_llm("gpt-3.5-turbo")

    def _build_faiss_index(self):
        if self._faiss_index is None:
            texts = reduce(
                lambda x, y: x + y, [doc["texts"] for doc in self.docs.values()], []
            )
            metadatas = reduce(
                lambda x, y: x + y, [doc["metadata"] for doc in self.docs.values()], []
            )
            self._faiss_index = FAISS.from_texts(
                texts, OpenAIEmbeddings(), metadatas=metadatas
            )

    async def get_evidence(
            self,
            answer: Answer,
            k: int = 3,
            max_sources: int = 5,
            marginal_relevance: bool = True,
            key_filter: Optional[List[str]] = None,
    ) -> str:
        if self._faiss_index is None:
            self._build_faiss_index()

        # perform vectorsearch
        _k = k

        if key_filter is not None:
            _k = k * 10  # heuristic

        docs = self.vector_search(answer, _k, marginal_relevance=marginal_relevance)
        # get summaries
        print(f'OpenAI summarization started at {datetime.now().time().strftime("%X")}')
        print(f'Summarizing {len(docs)} docs.')
        llm_summaries = await async_get_summaries(docs, answer.question)
        print(f'OpenAI summarization finished at {datetime.now().time().strftime("%X")}')

        # Grab the information from the nearest neigbors metadata
        for i, doc in enumerate(docs):
            if key_filter is not None and doc.metadata["dockey"] not in key_filter:
                continue

            c = (
                doc.metadata["key"],
                doc.metadata["citation"],
                llm_summaries[i],
                doc.page_content
            )

            if "Not applicable" not in c[2]:
                answer.contexts[doc.metadata['unique_id']] = c

            if len(answer.contexts) == max_sources:
                break

        # Create context_str which has relevant sources and their citation
        # will be fed into LLM for final answer
        context_str = "\n\n".join(
            [f"{k}: {s}" for k, c, s, t in answer.contexts.values() if "Not applicable" not in s]
        )
        valid_keys = [k for k, c, s, t in answer.contexts.values() if "Not applicable" not in s]
        if len(valid_keys) > 0:
            context_str += "\n\nValid keys: " + ", ".join(valid_keys)
        answer.context = context_str

        return answer

    async def query(
            self,
            query: str,
            k: int = 10,
            max_sources: int = 10,
            length_prompt: str = "about 100 words",
            marginal_relevance: bool = True,
            embedding: Optional[List[float]] = None,
            vector_search_only: bool = False
    ):

        if k < max_sources:
            raise ValueError("k should be greater than max_sources")
        tokens = 0
        answer = Answer(query, question_embedding=embedding)

        if embedding is not None:
            answer.question_embedding = embedding
            answer.from_embed = True

        with get_openai_callback() as cb:
            answer = await self.get_evidence(
                answer,
                k=k,
                max_sources=max_sources,
                marginal_relevance=marginal_relevance,
            )
            tokens += cb.total_tokens

        context_str, references_info = answer.context, answer.contexts

        bib = dict()
        passages = dict()

        if len(context_str) < 10:
            answer_text = (
                "I cannot answer this question due to insufficient information."
            )
            formatted_answer = f"Question: {query}\n\n{answer_text}\n"
            answer.formatted_answer = formatted_answer  # polished answer
        elif not vector_search_only:
            with get_openai_callback() as cb:
                answer_text = self.qa_chain.run(
                    question=query, context_str=context_str, length=length_prompt
                )
                tokens += cb.total_tokens

            # it still happens lol
            if "(Foo2012)" in answer_text:
                answer_text = answer_text.replace("(Foo2012)", "")

            for key, citation, summary, text in references_info.values():
                # do check for whole key (so we don't catch Callahan2019a with Callahan2019)
                skey = key.split(" ")[0]
                if skey + " " in answer_text or skey + ")" or skey + "," in answer_text:
                    bib[skey] = citation
                    passages[key] = text

                bib_str = "\n\n".join(
                    [f"{i + 1}. ({k}): {c}" for i, (k, c) in enumerate(bib.items())]
                )
                formatted_answer = f"Question: {query}\n\n{answer_text}\n"
                if len(bib) > 0:
                    formatted_answer += f"\nReferences\n\n{bib_str}\n"

                answer.answer = answer_text  # LLM answer from context strings
                answer.formatted_answer = formatted_answer  # polished answer
                answer.references = bib_str  # string for bibliography
                answer.passages = passages  # text chunks chosen by LLM for answer
                answer.tokens = tokens  # Number of tokens to answer question

        return answer

    def vector_search(self, answer, _k, marginal_relevance=True):
        # want to work through indices but less k
        if marginal_relevance:
            if not answer.from_embed:
                docs = self._faiss_index.max_marginal_relevance_search(
                    answer.question, k=_k, fetch_k=5 * _k
                )
            else:
                docs = self._faiss_index.max_marginal_relevance_search_by_vector(
                    answer.question_embedding, k=_k, fetch_k=5 * _k
                )
        else:
            docs = self._faiss_index.similarity_search(
                answer.question, k=_k, fetch_k=5 * _k
            )

        return docs


import asyncio


async def async_openAI_call(doc, question, n):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    summary_chain = make_chain(prompt=summary_prompt, llm=llm)
    summary = await summary_chain.arun(
        question=question,
        context_str=doc.page_content,
        citation=doc.metadata["citation"],
    )

    return summary


async def async_get_summaries(docs, question):
    coroutines = [async_openAI_call(doc, question, i) for i, doc in enumerate(docs)]

    summaries = await asyncio.gather(*coroutines)

    return summaries
