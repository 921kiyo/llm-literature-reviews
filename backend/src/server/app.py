from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import arxiv

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
        output["url"] = url
        outputs.append(output)
    return outputs

def search_term_refiner(messages) -> list:
    
    system = "You are a arxiv api query master and you will receive questions and try to generate several better search terms queries based on the questions.\
        The possible results will be displayed in a list. The list will need to follow the exact format as the following and only return the list:\
                If the search terms are not well-defined in the scientific community, return an empty list. \
                 Question: How can carbon nanotubes be manufactured at a large scale \
                        Answer: ['all:carbon nanotubes+AND+all:manufacturing', 'all:carbon nanotubes+AND+all:large-scale production']"
    message =[{"role": "system","content": system}]

    message.append(
            {"role": "user",
            "content": "Questions: {}, \n Answer:".format("What is stable diffusion model?")}
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

@app.post("/search/")
async def search_paper(message: SearchItem):
    message_list = search_term_refiner(message.search_term)
    #TODO: parallelize the following arxiv api to return the search result at the same time.
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

    # relevant_documents = {url: parsed_arxiv_results[url] for url in nearest_neighbors}

    # relevant_pdfs, answers = qa_pdf(question=message.search_term,
    #                        k=20,
    #                        parsed_arxiv_results=relevant_documents,
    #                        question_embeddings=question_embeddings)

    return {"question": asb_answers[0].question,
            "answer": asb_answers[0].answer,
            "context": asb_answers[0].context,
            "contexts": asb_answers[0].contexts,
            "references": clean_ref}


