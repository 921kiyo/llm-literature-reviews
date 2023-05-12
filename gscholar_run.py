import os

# Set directory for target documents
os.environ['ROOT_DIRECTORY'] = ROOT_DIRECTORY = 'backend/src/question_answer_pipeline/test'
import json
from backend.src.question_answer_pipeline.src.utils import from_pdfs_docstore, make_query
from backend.src.question_answer_pipeline.src.embedding import embed_questions
from dotenv import load_dotenv
from backend.src.question_answer_pipeline.src.utils import from_arxiv_docstore
import arxiv
from concurrent.futures import ThreadPoolExecutor
from serpapi import GoogleSearch

load_dotenv()

from backend.src.question_answer_pipeline.src.utils import qa_abstracts, qa_pdf, \
    parse_arxiv_json, download_pdfs_from_arxiv

if __name__ == '__main__':
    import argparse

    # Create the argument parser
    parser = argparse.ArgumentParser(description='Test run the files with google scholar api.')

    # Add arguments
    parser.add_argument('search', type=str, help='Input file path')

    # Parse the arguments
    args = parser.parse_args()

    print('Search: {}'.format(args.search))


    # openai.api_key  = os.getenv('OPENAI_API_KEY')
    secret_api_key = os.getenv('SERPAI_API_KEY')
    def search_arxiv():
        search_results = arxiv.Search(
            query=args.search,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        ).results()
        return search_results
    # Process the arXiv search results here

    # Define a function for the second command
    def search_google_scholar():
        params = {
            "engine": "google_scholar",
            "q": args.search,
            "hl": "en",
            "num": 5,
            "api_key": secret_api_key
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        return results

    # Create a ThreadPoolExecutor with maximum concurrent threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit the functions to the executor
        arxiv_future = executor.submit(search_arxiv)
        google_scholar_future = executor.submit(search_google_scholar)

        # Wait for the results
        arxiv_results = arxiv_future.result()
        google_scholar_results = google_scholar_future.result()
    print(arxiv_results)
    print()
    print('The following is the gscholoar returns result: \n')
    print(google_scholar_results)

    




