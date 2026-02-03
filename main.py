from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse

app = FastAPI()

class SearchReq(BaseModel):
    url: str
    top_pages: int = 20
    mode: str = "main_only"

@app.post("/search")
def search(req: SearchReq):
    return {
        "query_url": req.url,
        "mode": req.mode,
        "top_pages": req.top_pages,
        "results": [],
        "status": "connected"
    }

@app.post("/export_csv")
def export_csv(req: SearchReq):
    return {
        "ok": True,
        "download_url": "/download/result.csv"
    }

@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    return "<h1>Privacy Policy</h1><p>No personal data stored.</p>"
