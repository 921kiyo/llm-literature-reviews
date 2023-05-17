## How to run Frontend

1. `cd frontend`
2. Run `npm install`
3. Run `npm run dev`

## How to run FastAPI backend

1. `pip install -r requirements.txt`
2. `cd backend/src`
3. `uvicorn server.app:app --reload`
4. Go to `http://127.0.0.1:8000/docs` in your browser and you can make REST call (GET, POST) from the browser.

## Environment variables

```
export MODAL=false
export OPENAI_API_KEY=sk-XXX
export ROOT_DIRECTORY=local-directory
export COHERE_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
export SERPAPI_API_KEY=your_key
```

## The Docs Class

### Document Metadata

metadata is used to identify embeddings in the database.
Constructed as:

```
dict(
    citation=citation for document,
    dockey=document identifier,
    key=identifier for specific chunk (i.e. page in document")
)
example:
    pdf_metadata = dict(
                        citation= 'MLA citation',
                        dockey= 'Author, year',
                        key= 'Author, year. - Page 1'
                    )
```
