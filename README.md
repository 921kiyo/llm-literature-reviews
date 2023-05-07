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
```
