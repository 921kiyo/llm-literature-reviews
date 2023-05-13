from fastapi import FastAPI
from .schemas import SearchItem, Chat
from fastapi.middleware.cors import CORSMiddleware
import arxiv
from dotenv import load_dotenv
from tqdm import tqdm
import openai
import os

load_dotenv()
import datetime
from question_answer_pipeline.src.utils import qa_abstracts, qa_pdf, parse_arxiv_json, download_relevant_documents

app = FastAPI()

origins = [
    "http://192.168.211.:8000",
    "http://127.0.0.1:8000",
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


def parse_search_results(results):
    # TODO this is so hacky, so will fix this later
    import os

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

    return output


def get_references(parsed_arxiv_results, contexts):
    outputs = []
    for url in parsed_arxiv_results:
        contexts_key = os.path.split(url)[1]
        if contexts_key in contexts:
            output = {}
            output["title"] = parsed_arxiv_results[url]["title"]
            output["authors"] = parsed_arxiv_results[url]["authors"]
            output["journal"] = parsed_arxiv_results[url]["journal"]
            output["llm_summary"] = contexts[contexts_key][2]
            output["url"] = url
            outputs.append(output)
    return outputs


def search_term_refiner(search_question) -> list:
    openai.api_key = os.getenv('OPENAI_API_KEY')
    system = "You are a Google search master and you will receive a question and try to come up with a better search terms queries based on the question.\
        The possible search keywords will be displayed in a list. The list will need to follow the exact format as the following and only return the list:\
                If the search terms are not well-defined in the scientific community, return an empty list. \
                 Question: What is the current limitation of large language model? \
                        Answer: ['limitation', 'large language model']"

    message = [{"role": "system", "content": system}]

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


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/chat/")
async def ask_question(chat: Chat):
    relevant_documents = {chat.url: chat.parsed_arxiv_results[chat.url]}
    relevant_pdfs, relevant_answers = await qa_pdf(question=chat.question, k=20,
                                                   parsed_arxiv_results=relevant_documents)
    return {"answer": relevant_answers[0].answer}


@app.post("/search/")
async def search_paper(message: SearchItem):
    ## Hyperparameters
    max_results = 25
    documents_for_full_analysis = 10
    pdf_chunks_to_consider = 30

    refined_search_keywords = search_term_refiner(message.search_term)
    search_keyword = ' OR '.join(refined_search_keywords)
    print(search_keyword)
    search_results = arxiv.Search(
        query=search_keyword,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    ).results()

    search_results_list = parse_search_results(search_results)
    parsed_arxiv_results = parse_arxiv_json(search_results_list)

    if not parsed_arxiv_results:
        print(search_results)
        print('NO RESULTS')

    for key in parsed_arxiv_results:
        print(f'Raw results: {key}')

    start = datetime.datetime.now()
    print(start.strftime("%H:%M:%S"))
    nearest_neighbors, question_embeddings, asb_answers = await qa_abstracts(question=message.search_term, k=documents_for_full_analysis,
                                                                             parsed_arxiv_results=parsed_arxiv_results)
    end = datetime.datetime.now()
    print(end.strftime("%H:%M:%S"), f'elapsed (s): {(end - start).total_seconds():.3}')
    print('-' * 50)

    clean_ref = get_references(parsed_arxiv_results, asb_answers[0].contexts)

    if not nearest_neighbors:
        # TODO: error handling output
        print('Cannot answer your question.')
    else:
        # TODO: download papers here not at the beginning.
        print(f'Nearest Neighbors: {list(nearest_neighbors.keys())}')
        print('Getting Answer from PDFs')
        relevant_documents = {url: parsed_arxiv_results[url] for url in nearest_neighbors}

        download_relevant_documents(relevant_documents)

        print(f'{list(relevant_documents.keys())}')

        # relevant_pdfs = dict(url= (key, citation, llm_summary, text_chunk_from_pdf))
        print('-' * 50)
        start = datetime.datetime.now()
        print(start.strftime("%H:%M:%S"))
        relevant_pdfs, relevant_answers = await qa_pdf(question=message.search_term,
                                                       k=pdf_chunks_to_consider,
                                                       parsed_arxiv_results=relevant_documents,
                                                       question_embeddings=question_embeddings)
        end = datetime.datetime.now()
        print(end.strftime("%H:%M:%S"), f'elapsed (s): {(end - start).total_seconds():.3}')
        print('-' * 50)

        print(f'ABS KEYS: {list(asb_answers[0].contexts.keys())}', '\n',
              f'PDF KEYS: {list(relevant_answers[0].contexts.keys())}')

        output_obj = relevant_answers
        clean_ref = get_references(parsed_arxiv_results, output_obj[0].contexts)

    return {"question": output_obj[0].question,
            "answer": output_obj[0].answer,
            "context": output_obj[0].context,
            "contexts": output_obj[0].contexts,
            "references": clean_ref,
            "arxiv_results": parsed_arxiv_results}
