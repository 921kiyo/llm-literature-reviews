from fastapi import FastAPI
from .schemas import SearchItem, Chat
from fastapi.middleware.cors import CORSMiddleware
import arxiv
from dotenv import load_dotenv
from tqdm import tqdm
load_dotenv()

from question_answer_pipeline.src.utils import qa_abstracts, qa_pdf, parse_arxiv_json

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


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/chat/")
async def ask_question(chat: Chat):
    relevant_documents = {chat.url: chat.parsed_arxiv_results[chat.url]}
    relevant_pdfs, relevant_answers = await qa_pdf(question=chat.question, k=20, parsed_arxiv_results=relevant_documents)
    return {"answer": relevant_answers[0].answer}


@app.post("/search/")
async def search_paper(message: SearchItem):
    search_results = arxiv.Search(
        query = message.search_term,
        max_results = 2,
        sort_by = arxiv.SortCriterion.Relevance,
        sort_order = arxiv.SortOrder.Descending
    ).results()
    if not search_results:
        print(search_results)
        print('NO RESULTS')
    search_results_list = parse_search_results(search_results)
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
        relevant_pdfs, relevant_answers = await qa_pdf(question=question, k=5, parsed_arxiv_results=relevant_documents, question_embeddings=question_embeddings)

    return {"question": asb_answers[0].question,
            "answer": asb_answers[0].answer,
            "context": asb_answers[0].context,
            "contexts": asb_answers[0].contexts,
            "references": clean_ref,
            "arxiv_results": parsed_arxiv_results}


