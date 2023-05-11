from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import arxiv
from dotenv import load_dotenv
import os
from tqdm import tqdm
import openai
load_dotenv()

from question_answer_pipeline.src.utils import qa_abstracts, qa_pdf, parse_arxiv_json

app = FastAPI()

origins = [
   "http://192.168.211.:8000",
   "http://localhost",
   "http://localhost:3000",
]

app.add_middleware(
   CORSMiddleware,
   allow_origins=origins,
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)

class SearchItem(BaseModel):
    search_term: str

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
            "authors": [{'name': author.name} for author in result.authors]
        })
        filename = result.entry_id.split('/')[-1]+'.pdf'
        filepath = os.path.join(pdf_dir, filename)
        print(f'PDFdir: {pdf_dir}, filename: {filename}, filepath: {filepath}')
        if not os.path.exists(filepath):
            result.download_pdf(dirpath=pdf_dir, filename=filename)
    return output

def get_references(parsed_arxiv_results, contexts):
    outputs = []
    for url in contexts.keys():
        output = {}
        output["title"] = parsed_arxiv_results[url]["title"]
        output["authors"] = parsed_arxiv_results[url]["authors"]
        output["journal"] = parsed_arxiv_results[url]["journal"]
        output["llm_summary"] = contexts[url][2]
        output["url"] = url
        outputs.append(output)
    return outputs

def search_term_refiner(search_question) -> list:
    openai.api_key  = os.getenv('OPENAI_API_KEY')
    system = "You are a arxiv api query master and you will receive questions and try to generate several better search terms queries based on the questions.\
        The possible results will be displayed in a list. The list will need to follow the exact format as the following and only return the list:\
                If the search terms are not well-defined in the scientific community, return an empty list. \
                 Question: How can carbon nanotubes be manufactured at a large scale \
                        Answer: ['all:carbon nanotubes+AND+all:manufacturing', 'all:carbon nanotubes+AND+all:large-scale production']"
    message =[{"role": "system","content": system}]

    message.append(
            {"role": "user",
            "content": "Questions: {}, \n Answer:".format(search_question)}
    )

    completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=message,
            temperature=0,
            ).choices[0].message["content"]
    output_queries = []
    new_result = completion.split("'")
    for idx, res in enumerate(new_result):
        if idx % 2 == 1:
                output_queries.append(res)
    return output_queries

def search_result_combiner(queries):
    import itertools
    search_result_list = []
    for term in queries:
        search_result_list.append(arxiv.Search(
                query = term,
                max_results = 5,
                sort_by = arxiv.SortCriterion.Relevance,
                sort_order = arxiv.SortOrder.Descending
            ).results())
    # Merge the results using itertools.chain()
    merged_results = itertools.chain.from_iterable(search_result_list)
    return merged_results


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/search/")
async def search_paper(message: SearchItem):
    search_results = search_term_refiner(message.search_term)
    # search_results = arxiv.Search(
    #     query = message.search_term,
    #     max_results = 5,
    #     sort_by = arxiv.SortCriterion.Relevance,
    #     sort_order = arxiv.SortOrder.Descending
    # ).results()
    if not search_results:
        print(search_results)
        print('NO RESULTS')
    all_search_results = search_result_combiner(search_results)
    search_results_list = parse_search_results(all_search_results)
    if not search_results_list:
        print('NO RESULTS')
    print(search_results_list)
    parsed_arxiv_results = parse_arxiv_json(search_results_list)

    for key in parsed_arxiv_results:
        print(f'Raw results: {key}')
        print(parsed_arxiv_results[key]['summary'])

    question = 'what is a neural network?'
    nearest_neighbors, question_embeddings, asb_answers = await qa_abstracts(question=question, k=5,
                                                                             parsed_arxiv_results=parsed_arxiv_results)

    clean_ref = get_references(parsed_arxiv_results, asb_answers[0].contexts)

    if not nearest_neighbors:
        print('Cannot answer your question.')
    else:
        print(f'Nearest Neighbors: {list(nearest_neighbors.keys())}')
        print('Getting Answer from PDFs')
        relevant_documents = {url: parsed_arxiv_results[url] for url in nearest_neighbors}
        print(f'{list(relevant_documents.keys())}')

        # relevant_pdfs = dict(url= (key, citation, llm_summary, text_chunk_from_pdf))
        relevant_pdfs, relevant_answers = await qa_pdf(question=message.search_term, k=5, parsed_arxiv_results=relevant_documents, question_embeddings=question_embeddings)

    return {"question": asb_answers[0].question,
            "answer": asb_answers[0].answer,
            "context": asb_answers[0].context,
            "contexts": asb_answers[0].contexts,
            "references": clean_ref}


