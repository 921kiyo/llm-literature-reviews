from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import arxiv
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
    hardcoded_search_term = "Stable diffusion model"
    
    # TODO: Do Arxiv API call and fetch the top 10 results
    search = arxiv.Search(
    query = hardcoded_search_term,
    max_results = 10,
    sort_by = arxiv.SortCriterion.Relevance,
    sort_order = arxiv.SortOrder.Descending
    )
    result_message = ""
    for result in search.results():
        result_message += result.title +' --- \n'
        result_message += result.summary +' --- \n'
    return result_message

