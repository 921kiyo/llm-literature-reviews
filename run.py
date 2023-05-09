import os

# Set directory for target documents
os.environ['ROOT_DIRECTORY'] = ROOT_DIRECTORY = 'backend/src/question_answer_pipeline/test'
import json
from backend.src.question_answer_pipeline.src.utils import from_pdfs_docstore, make_query
from backend.src.question_answer_pipeline.src.embedding import embed_questions
from dotenv import load_dotenv
from backend.src.question_answer_pipeline.src.utils import from_arxiv_docstore
import arxiv

load_dotenv()

from backend.src.question_answer_pipeline.src.utils import qa_abstracts, qa_pdf, \
    parse_arxiv_json, download_pdfs_from_arxiv

if __name__ == '__main__':
    import argparse

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Question and answer")
    # Define the command-line arguments using the add_argument() method
    parser.add_argument("question", type=str, help="Your name")
    parser.add_argument("search", type=str, help="Your name")

    args = parser.parse_args()
    print(f'Question: {args.question}')
    print(f'Search: {args.search}')

    # file = './backend/src/question_answer_pipeline/example_arxiv_result.json'
    # arxiv_results = json.load(open(file, 'r'))

    arxiv_results = arxiv.Search(
        query=args.search,
        max_results=2,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    ).results()

    outputs = []
    for result in arxiv_results:
        outputs.append({
            'published': str(result.published),
            "entry_id": result.entry_id,
            'summary': result.summary,
            'title': result.title,
            "authors": [{'name': author.name} for author in result.authors],
            'download_handle': result.download_pdf
        })

    # parse the arxiv results
    parsed_arxiv_results = parse_arxiv_json(outputs)
    for key in parsed_arxiv_results:
        print(f'Raw results: {key}')
        print(parsed_arxiv_results[key]['summary'])
    # dict(url=dict(summary: summary, citation: MLA formatted citation, key: author, year))

    # get nearest neigbors
    print('-' * 10)
    print('Grabbing nearest neighbors from abstracts')
    # nearest_neighbors = dict(url=(key, citation, LLM summary related to question, original_abstract))
    nearest_neighbors, question_embeddings = qa_abstracts(question=args.question, k=5,
                                                          parsed_arxiv_results=parsed_arxiv_results)

    if not nearest_neighbors:
        print('Cannot answer your question.')
    else:
        print(f'Nearest Neighbors: {list(nearest_neighbors.keys())}')
        print('\n', '-' * 10)
        relevant_arxiv_results = [output for output in outputs if output['entry_id'] in nearest_neighbors]
        download_pdfs_from_arxiv(relevant_arxiv_results=relevant_arxiv_results)

        # do question and answer on the pdfs
        print('\n', '-' * 10)
        print('Getting Answer from PDFs')
        relevant_documents = {url: parsed_arxiv_results[url] for url in nearest_neighbors}
        print(f'{list(relevant_documents.keys())}')
        # relevant_pdfs = dict(url= (key, citation, llm_summary, text_chunk_from_pdf))
        relevant_pdfs = qa_pdf(question=args.question, k=20, parsed_arxiv_results=relevant_documents,
                               question_embeddings=question_embeddings)