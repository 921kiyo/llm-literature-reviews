from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
app = FastAPI()

class SearchItem(BaseModel):
    search_term: str
    

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/search/")
async def search_paper(message: SearchItem):
    return message

    # try:
    #     return message
    # except TypeError:
    #     print("Wrong input data type.")
    

