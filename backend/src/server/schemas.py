from pydantic import BaseModel

class SearchItem(BaseModel):
    search_term: str

class Chat(BaseModel):
    question: str
    url: str
    parsed_arxiv_results: dict
