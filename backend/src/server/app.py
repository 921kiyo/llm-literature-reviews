from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import arxiv
import tempfile

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


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/search/")
async def search_paper(message: SearchItem):
    search_results = arxiv.Search(
        query = message.search_term,
        max_results = 5,
        sort_by = arxiv.SortCriterion.Relevance,
        sort_order = arxiv.SortOrder.Descending
    ).results()

    # TODO: try webscraping instead as it might be faster
    with tempfile.TemporaryDirectory() as tempdir:
        # TODO: Make this step parallel
        for result in search_results:
            result.download_pdf(dirpath=tempdir)

    return 0


