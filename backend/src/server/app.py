from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import arxiv
import tempfile

from question_answer_pipeline.src.utils import qa_abstracts

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
    output = []
    for result in results:
        output.append({
            'published': str(result.published),
            "entry_id": result.entry_id,
            'summary': result.summary,
            'title': result.title,
            "authors": [{'name': author.name} for author in result.authors]
        })
    return output

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/search/")
async def search_paper(message: SearchItem):
    search_results = arxiv.Search(
        query = message.search_term,
        max_results = 1,
        sort_by = arxiv.SortCriterion.Relevance,
        sort_order = arxiv.SortOrder.Descending
    ).results()
    search_results_list = parse_search_results(search_results)
    qa_abstracts(message.search_term, search_results_list)

    # TODO: try webscraping instead as it might be faster
    with tempfile.TemporaryDirectory() as tempdir:
        # TODO: Make this step parallel
        for result in search_results:
            result.download_pdf(dirpath=tempdir)

    return 0


