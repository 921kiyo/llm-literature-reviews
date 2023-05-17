import os

# Set directory for target documents
os.environ['ROOT_DIRECTORY'] = ROOT_DIRECTORY = 'backend/src/question_answer_pipeline/test'
import json
from dotenv import load_dotenv
import arxiv
from concurrent.futures import ThreadPoolExecutor
from serpapi import GoogleSearch
import time
from tqdm import tqdm

load_dotenv()

from backend.src.question_answer_pipeline.src.utils import qa_pdf, \
    parse_arxiv_json, download_pdfs_from_arxiv, download_relevant_documents

def cohere_rerank(question , top_k, parsed_arxiv_results):
    import cohere
    secret_api = os.getenv('COHERE_API_KEY')
    co = cohere.Client(secret_api)
    url_arxiv = {}
    for url in parsed_arxiv_results:
        url_arxiv[url] = parsed_arxiv_results[url]["title"] + '  ---  ' + parsed_arxiv_results[url]["summary"]
    mapping_url_arxiv = list(url_arxiv.keys())
    arxiv_content = list(url_arxiv.values())
    results = co.rerank(model="rerank-english-v2.0", query=question, documents=arxiv_content, top_n=top_k)
    output_dic = {}
    for obj in results:
        url = mapping_url_arxiv[obj.index]
        content = parsed_arxiv_results[url]
        output_dic[url] = content
    return output_dic

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
                    url_parsed_json[unique_id] = {'summary': summary,'citation': citation, 'key': key, "title": title, 
                                                  "authors": author, "journal": journal,'download_handle': None,
                                                   'unique_id': unique_id, 'From_Arxiv': False, 'download_url': link}
    return url_parsed_json

def parse_search_results(results):
    # TODO this is so hacky, so will fix this later
    import os
    pdf_dir = os.path.join(os.getenv("ROOT_DIRECTORY"), "pdfs")
    output = []
    # TODO: try webscraping instead as it might be faster
    # TODO: Make this step parallel
    for result in tqdm(results):
        output.append({
            'published': str(result.published),
            "entry_id": result.entry_id,
            'summary': result.summary,
            'title': result.title,
            "authors": [{'name': author.name} for author in result.authors],
            'download_handle': result.download_pdf
        })
        # filename = result.entry_id.split('/')[-1]+'.pdf'
        # filepath = os.path.join(pdf_dir, filename)
        # print(f'PDFdir: {pdf_dir}, filename: {filename}, filepath: {filepath}')
        # if not os.path.exists(filepath):
        #     result.download_pdf(dirpath=pdf_dir, filename=filename)
    return output    

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
    
    def search_arxiv(search_term):
        search_results = arxiv.Search(
            query=search_term,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        ).results()
        return search_results
    # Process the arXiv search results here

    # Define a function for the second command
    def search_google_scholar(search_term):
        serpai_api_key = os.getenv('SERPAPI_API_KEY')
        params = {
            "engine": "google_scholar",
            "q": search_term,
            "hl": "en",
            "num": 10,
            "api_key": serpai_api_key
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        return results

    # Create a ThreadPoolExecutor with maximum concurrent threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit the functions to the executor
        arxiv_future = executor.submit(search_arxiv, args.search)
        google_scholar_future = executor.submit(search_google_scholar)

        # Wait for the results
        arxiv_results = arxiv_future.result()
        # google_scholar_results = google_scholar_future.result()
    # print(arxiv_results)
    print()
    print('The following is the gscholoar returns result: \n')
    # print(google_scholar_results)

    initial_results = parse_search_results(arxiv_results)
    assert initial_results is not None, f'Return No results'
    arxiv_results = parse_arxiv_json(initial_results)
    google_scholar_json = parse_gscholar_json(google_scholar_results)
    print(f'The google search results are: {list(google_scholar_json.keys())}')
    arxiv_results.update(google_scholar_json)

    # ##################################################
    # ## The following is the test run from run.py
    # ##################################################
    # start = time.perf_counter()
    # nearest_neighbors = cohere_rerank(question = args.search, top_k = 10, parsed_arxiv_results=arxiv_results)
    # end = time.perf_counter() - start
    # print(f'elapsed (s): {end:.3}')
    # print(f'Nearest Neighbors: {list(nearest_neighbors.keys())}')
    # print('&' * 50)

    # parsed_arxiv_results = arxiv_results
    # if not nearest_neighbors:
    #     print('Cannot answer your question.')
    # else:
    #     print(f'Nearest Neighbors: {list(nearest_neighbors.keys())}')
    #     print('Getting Answer from PDFs')
    #     relevant_documents = {url: parsed_arxiv_results[url] for url in nearest_neighbors}
    #     download_relevant_documents(relevant_documents)
    #     print(f'{list(relevant_documents.keys())}')
    #     # print(f'{list(relevant_documents.values())}')

    #     # relevant_pdfs = dict(url= (key, citation, llm_summary, text_chunk_from_pdf))
    #     print('-' * 50)
    #     start = time.perf_counter()
    #     # print(start.strftime("%H:%M:%S"))
    #     relevant_pdfs, relevant_answers = qa_pdf(question=args.search, k=25, parsed_arxiv_results=relevant_documents)
    #     end = time.perf_counter - start
    #     print(f'The qa_pdf function runs for {end:.3} seconds')
    #     # print(end.strftime("%H:%M:%S"), f'elapsed (s): {(end - start).total_seconds():.3}')
    #     print('-' * 50)




    




