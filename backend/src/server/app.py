from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import arxiv

from question_answer_pipeline.src.utils import qa_abstracts, qa_pdf, parse_arxiv_json, download_pdfs_from_arxiv

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
    for result in results:
        output.append({
            'published': str(result.published),
            "entry_id": result.entry_id,
            'summary': result.summary,
            'title': result.title,
            "authors": [{'name': author.name} for author in result.authors]
        })
        filename = result.entry_id.split('/')[-1]+'.pdf'
        filepath = os.path.join(pdf_dir, filename)
        result.download_pdf(dirpath=pdf_dir, filename=filepath)
    return output

def get_references(parsed_arxiv_results, contexts):
    outputs = []
    for url in contexts.keys():
        output = {}
        output["title"] = parsed_arxiv_results[url]["title"]
        output["authors"] = parsed_arxiv_results[url]["authors"]
        output["journal"] = parsed_arxiv_results[url]["journal"]
        output["llm_summary"] = contexts[url][2]
        outputs.append(output)
    return outputs


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/search/")
async def search_paper(message: SearchItem):
    search_results = arxiv.Search(
        query = message.search_term,
        max_results = 2,
        sort_by = arxiv.SortCriterion.Relevance,
        sort_order = arxiv.SortOrder.Descending
    ).results()
    search_results_list = parse_search_results(search_results)

    parsed_arxiv_results = parse_arxiv_json(search_results_list)
    nearest_neighbors, question_embeddings, asb_answers = qa_abstracts(question=message.search_term,
                                                          k=5,
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
        relevant_pdfs = qa_pdf(question=message.search_term, k=20, parsed_arxiv_results=relevant_documents,
                               question_embeddings=question_embeddings)

    # relevant_pdfs, answers = qa_pdf(question=message.search_term,
    #                        k=20,
    #                        parsed_arxiv_results=relevant_documents,
    #                        question_embeddings=question_embeddings)

    return {"question": asb_answers[0].question,
            "answer": asb_answers[0].answer,
            "context": asb_answers[0].context,
            "contexts": asb_answers[0].contexts,
            "references": clean_ref}


