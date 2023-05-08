import os
from datetime import datetime
from InstructorEmbedding import INSTRUCTOR
import json
from tqdm import tqdm
import re
from ..qa_utils import readers, Docs
import pickle
from .embedding import embed_file_chunks, embed_questions
from langchain.chains import LLMChain
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessage
from langchain.chat_models import ChatOpenAI

ROOT_DIRECTORY = os.getenv('ROOT_DIRECTORY', '/test')

# pdf embedding related directories
FILE_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'pdfs')
INDEX_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'index')
PDF_EMB_DIR = os.path.join(ROOT_DIRECTORY, 'embeddings')
CITATIONS_FILE = os.path.join(ROOT_DIRECTORY, 'citations.json')

# abstracts embeddings related directories
ABSTRACTS_EMB_DIR = os.path.join(ROOT_DIRECTORY, 'abstract_embeddings')

# save docs file related directory (might not be needed for this application)
DOCS_FILE = os.path.join(ROOT_DIRECTORY, 'pdf_docs')


def qa_pdf(question, k, parsed_arxiv_results, question_embeddings=None):
    """
    qa on pdf documents
    :param question_embeddings: embedded question
    :param k:
    :param parsed_arxiv_results:
    :param question:
    :return: nearest neighbor information: answer.contexts = dict(url= (key, citation, llm_summary, chunked_text))
    """
    # create a docstore that stays updated with the filesystem
    # it is rebuilt if pdfs are deleted and items are added when new files are detected
    docs = from_pdfs_docstore(parsed_arxiv_results)

    queries = [question]

    if question_embeddings is None:
        print('embedding questions')
        question_embeddings = embed_questions(queries, use_modal=os.environ['MODAL'])

    print('getting answers')
    answers = make_query(docs, queries, question_embeddings, k=k, vector_search_only=False)

    for answer in answers:
        print('-' * 20)
        print('Answer from pdfs:')
        print(answer.formatted_answer)

    return answer.contexts, answers


def qa_abstracts(question, k, parsed_arxiv_results=None):
    """

    :param parsed_arxiv_results: dict(url=dict(summary: summary, citation: MLA formatted citation, key: author, year))
    :param k: number of documents to retrieve from semantic search
    :param question: user query
    :return:
        nearest neighbors as dict(url=(key, citation, LLM summary related to question, original_text))
        question_embedding: embedded question
    """
    docs = from_arxiv_docstore(parsed_arxiv_results)

    queries = [question]

    print('embedding question')
    question_embeddings = embed_questions(queries, use_modal=os.environ['MODAL'])

    print('getting answers')
    answers = make_query(docs, queries, question_embeddings, k=k, vector_search_only=False)

    for answer in answers:
        # answer.contexts = dict(url=(key, citation, LLM summary related to question, original_text))
        print('-' * 50)
        print('Answer from abstracts:')
        print(answer.formatted_answer)

    return answer.contexts, question_embeddings, answers


def parse_arxiv_json(arxiv_results):
    """
    Retrieves
    1. summary
    2. citation
    3. key used for inline citation.

    :param arxiv_results: arxiv JSON
    :return: dict. key=entry_id, val=dict(summary: summary, citation: MLA formatted citation, key: author, year)
    """
    from datetime import datetime

    mla_months = {
        1: "Jan.",
        2: "Feb.",
        3: "Mar.",
        4: "Apr.",
        5: "May",
        6: "June",
        7: "July",
        8: "Aug.",
        9: "Sept.",
        10: "Oct.",
        11: "Nov.",
        12: "Dec."
    }

    source = "arXiv. "

    url_parsed_json = {}

    for arxiv_res in arxiv_results:
        authors_list = [a['name'] for a in arxiv_res['authors']]
        authors = ', '.join(authors_list) + '. '
        title = arxiv_res['title'] + '. '
        published_date = datetime.fromisoformat(arxiv_res['published'])
        year = str(published_date.year)
        published_date = mla_months[published_date.month] + ', ' + year + '. '
        url = arxiv_res['entry_id']

        citation = authors + title + source + url.split('/')[-1] + '. ' + published_date + url
        key = f"{authors_list[0]}, {year}"

        summary = arxiv_res['summary']

        url_parsed_json[url] = {'summary': summary, 'citation': citation, 'key': key}

    return url_parsed_json


def embed_abstracts(parsed_arxiv_results):
    """

    :param parsed_arxiv_results: dictionary keys: entry_ids, values: dict that describes document
    :return:
    """
    os.makedirs(ABSTRACTS_EMB_DIR, exist_ok=True)
    existing_embeddings = set([i[:-4] for i in os.listdir(ABSTRACTS_EMB_DIR)])

    arxiv_entries = set([i.split('/')[-1] for i in parsed_arxiv_results.keys()])
    new_embeddings = arxiv_entries - existing_embeddings
    print(f'Embeddings to process: {new_embeddings}')

    to_process = {k: v for k, v in parsed_arxiv_results.items() if k.split('/')[-1] in new_embeddings}

    for entry_id, doc_info in tqdm(to_process.items()):
        print(f'Processing: {entry_id}')

        unique_id = entry_id
        citation = doc_info['citation']
        key = doc_info['key']
        summary = [doc_info['summary']]
        metadata = [dict(
            unique_id=unique_id,
            citation=citation,
            dockey=key,
            key=f'abstract_{key}'
        )]

        file_embeddings, num_tokens = embed_file_chunks(summary, use_modal=os.environ['MODAL'])

        # save text chunks, file embeddings, metadatas, and num_tokens for each
        save_dict = [summary, file_embeddings, metadata, num_tokens]

        path = os.path.join(ABSTRACTS_EMB_DIR, entry_id.split('/')[-1] + '.pkl')
        print(f'Saving abstract embeddings to path: {path}')

        # save embeddings
        with open(path, 'wb') as fp:
            pickle.dump(save_dict, fp)


def create_docs(relevant_documents, dir):
    """

    :param relevant_documents: unique id for documents from search
    :return:
    """
    print('Building DOCS')
    docs = Docs()

    for no_files, f in enumerate(relevant_documents):
        file_path = os.path.join(dir, f)

        with open(file_path, 'rb') as fb:
            processed_file = pickle.load(fb)

        filename = f[:-4]

        # parse processed file
        summary = processed_file[0]
        file_embeddings = processed_file[1]
        metadata = processed_file[2]  # dict(citation=citation, dockey= key, key=f"{key} pages {pg}",)

        # add to doc class
        docs.add_from_embeddings(path=filename,
                                 texts=summary,
                                 text_embeddings=file_embeddings,
                                 metadatas=metadata)

    print(f'added {no_files + 1} files to docs')

    return docs


def from_arxiv_docstore(parsed_arxiv_results):
    """

    :param parsed_arxiv_results: parsed json response from arxiv search
    :return: Docs class
    """

    # embed file summaries
    embed_abstracts(parsed_arxiv_results)

    # create docs class for vector search
    # embedding filenames for relevant documents
    rel_docs = [d.split('/')[-1] + '.pkl' for d in parsed_arxiv_results.keys()]
    docs = create_docs(rel_docs, dir=ABSTRACTS_EMB_DIR)

    return docs


def embed_pdf_files(parsed_arxiv_results):
    """

    :param parsed_arxiv_results:
    :return:
    """
    # directory to save embeddings
    os.makedirs(PDF_EMB_DIR, exist_ok=True)
    existing_embeddings = set([i[:-4] for i in os.listdir(PDF_EMB_DIR)])

    arxiv_entries = set([i.split('/')[-1] for i in parsed_arxiv_results.keys()])
    new_embeddings = arxiv_entries - existing_embeddings
    print(f'Embeddings to process: {new_embeddings}')

    to_process = {k: v for k, v in parsed_arxiv_results.items() if k.split('/')[-1] in new_embeddings}

    # read pdf, embed chunks
    parse_pdf = readers.parse_pdf
    for entry_id, doc_info in tqdm(to_process.items()):
        # get file path for file f
        f = entry_id.split('/')[-1] + '.pdf'
        f_path = os.path.join(FILE_DIRECTORY, f)

        print(f'Reading and Embedding: {f} at {f_path}')

        citation = doc_info['citation']
        key = doc_info['key']

        # get texts (splits) and metadata (citation, key, key_with_page)

        splits, metadatas = parse_pdf(f_path, key=key, citation=citation, chunk_chars=1100, overlap=100)

        file_embeddings, num_tokens = embed_file_chunks(splits, use_modal=os.environ['MODAL'])

        # save text chunks, file embeddings, metadatas, and num_tokens for each
        save_dict = [splits, file_embeddings, metadatas, num_tokens]

        path = os.path.join(PDF_EMB_DIR, f[:-3] + 'pkl')

        print(f'Saving pdf embeddings to path: {path}')
        # save embeddings
        with open(path, 'wb') as fp:
            pickle.dump(save_dict, fp)


def from_pdfs_docstore(parsed_arxiv_results):
    """
    Build from local directory of pdfs. For reference from PubFind, might not be needed here.

    :param parsed_arxiv_results: parsed json response from arxiv search
    :return: docs class
    """

    embed_pdf_files(parsed_arxiv_results)

    # create docs class for vector search
    # embedding filenames for relevant documents
    rel_docs = [d.split('/')[-1] + '.pkl' for d in parsed_arxiv_results.keys()]
    docs = create_docs(rel_docs, dir=PDF_EMB_DIR)

    return docs


def make_query(docs, queries, question_embeddings, k=5, vector_search_only=False):
    """

    :param docs:
    :param queries:
    :param question_embeddings:
    :param k:
    :param vector_search_only:
    :return: Answer.contexts: Contains results from nearest neighbors search:
                                dict(url=(key, citation, summary, chunked_text))
    """
    answers = []
    length_prompt = 'about 50 words'
    for query, embedding in zip(queries, question_embeddings):
        answers.append(docs.query(query,
                                  embedding=embedding,
                                  length_prompt=length_prompt,
                                  k=k,
                                  vector_search_only=vector_search_only)
                       )

    return answers


#####################################
# Possibly not needed
#####################################
def check_citations(files_in_directory):
    if os.path.exists(CITATIONS_FILE):
        path_citations_map = json.load(open(CITATIONS_FILE, 'r'))
    else:
        path_citations_map = {}

    files_to_get_citations = [os.path.join(FILE_DIRECTORY, file)
                              for file in files_in_directory
                              if file not in path_citations_map and file[-3:] == 'pdf']

    print(f'files_to_cite: {files_to_get_citations}')
    path_citations_map = get_citations(files_to_get_citations)

    print(path_citations_map)


def get_citations(list_of_filenames):
    """

    :param list_of_filenames:
    :return: filename_citation: dict(filename=citation)
    """
    system_message = SystemMessage(content="You are a scholarly researcher that answers in an unbiased, scholarly tone."
                                           "You sometimes refuse to answer if there is insufficient information.")

    citation_prompt = HumanMessagePromptTemplate.from_template(
        "Return a possible citation for the following text. Do not include URLs. "
        "Citation should be in MLA format. Do not summarize"
        "the text. Only return the citation with the DOI.\n"

        "text: {text}\n\n"
        "Citation:"
        "If a citation cannot be determined from the text return None."
    )
    llm = ChatOpenAI(temperature=0.1, max_tokens=512, model_name='gpt-3.5-turbo')

    chat_prompt = ChatPromptTemplate.from_messages([system_message, citation_prompt])
    cite_chain = LLMChain(prompt=chat_prompt, llm=llm)

    # peak first chunk
    filename_citation = {}
    for path in list_of_filenames:
        texts, _ = readers.parse_pdf(path, "", "", chunk_chars=5000, peak=True)
        citation = cite_chain.run(text=texts)

        if len(citation) < 3 or "Unknown" in citation or "insufficient" in citation:
            citation = None

        filename_citation[os.path.split(path)[1]] = citation

    # save citations.json
    with open(CITATIONS_FILE, 'w') as f:
        json.dump(filename_citation, f, indent=4)

    return filename_citation


def compare_object_with_dir(docs):
    """
    return set of filenames without the extension
    """
    # check if there are files missing from docs
    files_in_docs = set(docs.docs)
    embeddings_in_directory = set([f[:-4] for f in os.listdir(PDF_EMB_DIR) if f[-3:] == 'pkl'])

    return embeddings_in_directory.difference(files_in_docs)


def grab_key(citation):
    # get key for document
    try:
        author = re.search(r"([A-Z][a-z]+)", citation).group(1)
    except AttributeError:
        # panicking - no word??
        raise ValueError(
            f"Could not parse key from citation {citation}. Consider just passing key explicitly - e.g. docs.py (path, citation, key='mykey')"
        )
    try:
        year = re.search(r"(\d{4})", citation).group(1)
        if 1900 < int(year) < 2030:
            year = year
        else:
            year = ""
    except AttributeError:
        year = ""

    return f"{author}{year}"


def files_for_search(file_directory, delete_remove=True):
    """

    :param file_directory:
    :param delete_remove:
    :return:
        files_not_embedded: str (filename)
        files_removed: bool
    """
    # create embeddings directory if needed
    emb_dir = PDF_EMB_DIR
    if not os.path.exists(emb_dir):
        os.makedirs(emb_dir)

    files = [f[:-4] for f in os.listdir(file_directory)
             if f[-3:] == 'pdf']

    embedded_files = [f[:-4] for f in os.listdir(emb_dir) if f[-3:] == 'pkl']

    files_not_embedded = [f + '.pdf' for f in files if f not in embedded_files]

    # check if pdfs were removed. If so then we have to remove the embedding file as well
    files_embedded_but_no_longer_present = [f + '.pkl' for f in embedded_files if f not in files]

    files_removed = False
    if delete_remove and files_embedded_but_no_longer_present:
        files_removed = True
        print('Deleting')
        print(files_embedded_but_no_longer_present)
        for f in files_embedded_but_no_longer_present:
            remove_path = os.path.join(PDF_EMB_DIR, f)
            os.remove(remove_path)

    return files_not_embedded, files_removed
