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




def parse_gscholar_json(search_results):
    results = search_results['organic_results']
    url_parsed_json = {}
    for result in results:
        if 'resources' in result.keys(): 
            title = result['title']
            for key, item in list(result['publication_info'].items()):
                if key == 'summary':
                    newItem = item.split('-')
                    authors = newItem[0].split(', ')
                    author = authors[0].rstrip()
                    journal = newItem[-1].lstrip()
                    year = newItem[1].split(', ')[-1].rstrip()
                    key = f"{author}, {year}"
                    citation = author + '. ' + title + '. ' + journal + '. ' + year + '. ' 
                    unique_id = result['result_id']
                    link = result['resources'][0]['link']
                    summary = result['snippet']
                    url_parsed_json[unique_id] = {'unique_id': unique_id, 'download_link': link, 'summary': summary, 'citation': citation, 'key': key, "title": title, "authors": author, "journal": journal}
    print(url_parsed_json)                    
    return url_parsed_json

    

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
    serpai_api_key = os.getenv('SERPAI_API_KEY')
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
            "api_key": serpai_api_key
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

    arxiv_results = parse_arxiv_json(arxiv_results)
    google_scholar_json = parse_gscholar_json(google_scholar_results)
    arxiv_results.update(google_scholar_json)

    ##################################################
    ## The following is the test run from run.py
    ##################################################
    start = datetime.datetime.now()
    print(start.strftime("%H:%M:%S"))
    nearest_neighbors = cohere_rerank(question = message.search_term, top_k = 10, parsed_arxiv_results=parsed_arxiv_results)
    end = datetime.datetime.now()
    print(end.strftime("%H:%M:%S"), f'elapsed (s): {(end - start).total_seconds():.3}')
    print('-' * 50)

    if not nearest_neighbors:
        print('Cannot answer your question.')
    else:
        print(f'Nearest Neighbors: {list(nearest_neighbors.keys())}')
        print('Getting Answer from PDFs')
        relevant_documents = {url: parsed_arxiv_results[url] for url in nearest_neighbors}
        download_relevant_documents(relevant_documents)
        print(f'{list(relevant_documents.keys())}')

        # relevant_pdfs = dict(url= (key, citation, llm_summary, text_chunk_from_pdf))
        print('-' * 50)
        start = datetime.datetime.now()
        print(start.strftime("%H:%M:%S"))
        relevant_pdfs, relevant_answers = qa_pdf(question=message.search_term, k=25, parsed_arxiv_results=relevant_documents)
        end = datetime.datetime.now()
        print(end.strftime("%H:%M:%S"), f'elapsed (s): {(end - start).total_seconds():.3}')
        print('-' * 50)




    




