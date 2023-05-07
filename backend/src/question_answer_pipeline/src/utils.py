import os
from datetime import datetime
from InstructorEmbedding import INSTRUCTOR
import json
from tqdm import tqdm
import re
from ..qa_utils import readers, Docs
import pickle
from .embedding import embed_file_chunks
from langchain.chains import LLMChain
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessage
from langchain.chat_models import ChatOpenAI

ROOT_DIRECTORY = os.getenv('ROOT_DIRECTORY', '/test')

FILE_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'pdfs')
ABSTRACTS_EMB_DIR = os.path.join(ROOT_DIRECTORY, 'abstract_embeddings')
EMB_DIR = os.path.join(ROOT_DIRECTORY, 'embeddings')
CITATIONS_FILE = os.path.join(ROOT_DIRECTORY, 'citations.json')
DOCS_FILE = os.path.join(ROOT_DIRECTORY, 'docs')
INDEX_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'index')


def get_model():
    model_root = 'embedding_models'
    model = INSTRUCTOR('hkunlp/instructor-xl', cache_folder=model_root)

    return model


def parse_arxiv_json(arxiv_results):
    """
    Retrieves
    1. summary
    2. citation
    3. key used for inline citation.

    :param arXiv_results: arxiv JSON
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

    path_citations_map = {}

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

        path_citations_map[url] = {'summary': summary, 'citation': citation, 'key': key}

    return path_citations_map


def embed_abstracts(parsed_arxiv_results):
    os.makedirs(ABSTRACTS_EMB_DIR, exist_ok=True)
    existing_embeddings = set([i[:-4] for i in os.listdir(ABSTRACTS_EMB_DIR)])
    print(f'existing_embeddings: {existing_embeddings}')

    arxiv_entries = set([i.split('/')[-1] for i in parsed_arxiv_results.keys()])
    new_embeddings = arxiv_entries - existing_embeddings
    print(f'Embeddings to process: {new_embeddings}')
    to_process = {k: v for k, v in parsed_arxiv_results.items() if k.split('/')[-1] in new_embeddings}
    print(f'Embeddings to process: {to_process}')

    for entry_id, doc_info in tqdm(to_process.items()):

        print(f'Processing: {entry_id}')

        citation = doc_info['citation']
        key = doc_info['key']
        summary = [doc_info['summary']]

        file_embeddings, num_tokens = embed_file_chunks(summary, use_modal=os.environ['MODAL'])
        metadata = [dict(
            citation=citation,
            dockey=key,
            key=f'abstract_{key}'
        )]
        # save text chunks, file embeddings, metadatas, and num_tokens for each
        save_dict = [summary, file_embeddings, metadata, num_tokens]

        path = os.path.join(ABSTRACTS_EMB_DIR, entry_id.split('/')[-1] + '.pkl')
        print(f'Saving abstract embeddings to path: {path}')

        # save embeddings
        with open(path, 'wb') as fp:
            pickle.dump(save_dict, fp)


def create_abstract_docs(docs):
    print('Building DOCS')

    for no_files, f in enumerate(os.listdir(ABSTRACTS_EMB_DIR)):
        file_path = os.path.join(ABSTRACTS_EMB_DIR, f)

        with open(file_path, 'rb') as fb:
            processed_file = pickle.load(fb)

        filename = f[:-4]
        print(f'Adding: {filename}')

        # parse processed file
        summary = processed_file[0]
        file_embeddings = processed_file[1]
        metadata = processed_file[2]  # dict(citation=citation, dockey= key, key=f"{key} pages {pg}",)
        print('-'*10)
        print(metadata)
        # add to doc class
        docs.add_from_embeddings(path=filename,
                                 texts=summary,
                                 text_embeddings=file_embeddings,
                                 metadatas=metadata)
    print(f'added: {no_files + 1}')

    with open(DOCS_FILE, 'wb') as fb:
        pickle.dump(docs, fb)


def from_arxiv_docstore(arxiv_results):
    """

    :param arxiv_result: json response from arxiv search
    :return: Docs class
    """
    # compare pdf files with embedding pkl files. If they don't match we can add or remove files
    parsed_results = parse_arxiv_json(arxiv_results)

    # embed file summaries
    embed_abstracts(parsed_results)

    # create docs class for vector search
    docs = Docs(index_path=INDEX_DIRECTORY)
    create_abstract_docs(docs)

    return docs


def files_for_search(file_directory, delete_remove=True):
    # create embeddings directory if needed
    emb_dir = EMB_DIR
    if not os.path.exists(emb_dir):
        os.makedirs(emb_dir)

    files = [f[:-4] for f in os.listdir(file_directory)
             if f[-3:] == 'pdf' and 'cover' not in f.lower()]

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
            remove_path = os.path.join(EMB_DIR, f)
            os.remove(remove_path)
    # print(files_embedded_but_no_longer_present)
    # print(files)
    # print('-'*50)
    # print(embedded_files)
    # print(len(files_not_embedded))
    return files_not_embedded, files_removed


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


def embed_files(files_to_embed):
    os.makedirs(EMB_DIR, exist_ok=True)

    if os.path.exists(CITATIONS_FILE):
        path_citations_map = json.load(open(CITATIONS_FILE, 'r'))
    else:
        raise FileNotFoundError('Citations File Does Not Exist. Rebuild DOCSTORE')

    print(path_citations_map)
    # get model
    parse_pdf = readers.parse_pdf

    for f in tqdm(files_to_embed):
        print(f'Processing: {f}')
        citation = path_citations_map[f]

        key = grab_key(citation)

        # get texts (splits) and metadata (citation, key, key_with_page)
        f_path = os.path.join(FILE_DIRECTORY, f)

        splits, metadatas = parse_pdf(f_path, key=key, citation=citation, chunk_chars=1200, overlap=100)

        file_embeddings, num_tokens = embed_file_chunks(splits, use_modal=os.environ['MODAL'])

        # save text chunks, file embeddings, metadatas, and num_tokens for each
        save_dict = [splits, file_embeddings, metadatas, num_tokens]

        path = os.path.join(EMB_DIR, f[:-3] + 'pkl')
        print(f'Saving embeddings to path: {path}')
        # save embeddings
        with open(path, 'wb') as fp:
            pickle.dump(save_dict, fp)


def compare_object_with_dir(docs):
    """
    return set of filenames without the extension
    """
    # check if there are files missing from docs
    files_in_docs = set(docs.docs)
    embeddings_in_directory = set([f[:-4] for f in os.listdir(EMB_DIR) if f[-3:] == 'pkl'])

    return embeddings_in_directory.difference(files_in_docs)


def create_and_update_docs(docs):
    """
    compare Docs contents with directory
    - add embeddings to docstore if needed
    """
    embedding_files_to_add = compare_object_with_dir(docs)
    citations = json.load(open(CITATIONS_FILE, 'r'))

    if len(embedding_files_to_add) == 0:
        print('Docstore is up to date')
        return
    else:
        # need to compare doc keys and set made from embedding directory
        print('Adding embedding_files_to_add')
        for no_files, f in enumerate(embedding_files_to_add):
            file_path = os.path.join(EMB_DIR, f + '.pkl')

            with open(file_path, 'rb') as fb:
                processed_file = pickle.load(fb)

            filename_without_extension = f
            print(f'Adding: {filename_without_extension}')

            # parse processed file
            splits = processed_file[0]
            file_embeddings = processed_file[1]
            metadatas = processed_file[2]  # dict(citation=citation, dockey= key, key=f"{key} pages {pg}",)

            # Update citations with existing JSON file
            for metadata in metadatas:
                metadata['citation'] = citations[filename_without_extension + '.pdf']

            num_tokens = processed_file[3]

            if len(metadatas[-1].keys()) < 3:
                metadatas[-1] = metadatas[-2]

            # add to doc class
            docs.add_from_embeddings(path=filename_without_extension,
                                     texts=splits,
                                     text_embeddings=file_embeddings,
                                     metadatas=metadatas)
        print(f'added: {no_files + 1}')

        with open(DOCS_FILE, 'wb') as fb:
            pickle.dump(docs, fb)

        return


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


def initialize_docstore(force_rebuild=False):
    """
    Build from local directory of pdfs. For reference from PubFind, might not be needed here.

    :param force_rebuild: rebuild DOCS class
    :return: docs class
    """
    # compare pdf files with embedding pkl files. If they don't match we can add or remove files
    files_to_embed, files_removed_bool = files_for_search(FILE_DIRECTORY)

    check_citations(os.listdir(FILE_DIRECTORY))

    # embed files
    if files_to_embed:
        print('Embedding New Files')
        # grab model from local or download if needed

        embed_files(files_to_embed)

        # double check that everything is embedded
        # remaining_files = files_for_search(FILE_DIRECTORY)

        # there may be an issue here where pkl takes a while to save to filesystem and the program continues executing

    # generate Doc or load Doc
    # if doc exists, and we haven't removed any files load the doc object
    if os.path.exists(DOCS_FILE) and not files_removed_bool and not force_rebuild:
        print('Loading DOCS')
        with open(DOCS_FILE, 'rb') as fb:
            docs = pickle.load(fb)
    else:  # if doc doesn't exist, or we have removed files create a new instance of the object
        print('creating new DOCS')
        docs = Docs(index_path=INDEX_DIRECTORY)

    # add embeddings
    create_and_update_docs(docs)

    return docs


def update_filenames():
    for f in os.listdir(EMB_DIR):
        old_path = os.path.join(EMB_DIR, f)
        new_path = os.path.join(EMB_DIR, f[:-15] + '.pkl')
        # print(new_path)
        os.rename(old_path, new_path)


def get_answers(docs, queries, question_embeddings):
    answers = []
    length_prompt = 'about 50 words'
    for query, embedding in zip(queries, question_embeddings):
        answers.append(docs.query(query, embedding=embedding, length_prompt=length_prompt, k=5))

    return answers


def get_citations(list_of_filenames):
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
    path_citation = {}
    for path in list_of_filenames:
        texts, _ = readers.parse_pdf(path, "", "", chunk_chars=5000, peak=True)
        citation = cite_chain.run(text=texts)

        if len(citation) < 3 or "Unknown" in citation or "insufficient" in citation:
            citation = None

        path_citation[os.path.split(path)[1]] = citation

    # save citations.json
    with open(CITATIONS_FILE, 'w') as f:
        json.dump(path_citation, f, indent=4)

    return path_citation
