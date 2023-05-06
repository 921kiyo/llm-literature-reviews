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
    query = message.search_term,
    # query = hardcoded_search_term,
    max_results = 1, # TODD change to 100 later
    sort_by = arxiv.SortCriterion.Relevance,
    sort_order = arxiv.SortOrder.Descending
    )
    print(search.results())

    paper = next(arxiv.Search(id_list=["1605.08386v1"]).results())
    # Download the PDF to the PWD with a default filename.

    # 0. Make embedding for the question (Hector)
    # 1. Make embedding the list of 100 abstracts (Hector)
    # 2 do the k-nearst neighbors between question on the abstracts embeddings (from paper qb), get the top 10 papers (Hector)

    # Kiyo to find out download PDF logic (or faster way)
    # Yuan to do some research on embedding optimization with LLM (Cohere rerank)

    # 3.1 Use LLM to summarize each 10 abstracts and the question
    # 3.2 Download 10 papers, make embedding for each, and answer the question with LLM
    # 3.3 Pass the entire 10 papers to GPT-4 and answer the question

    # (Extension) 4. Let the users be able to interact with a particular paper

    # Think of how to display the results in the frontend UI

    return search.results()


