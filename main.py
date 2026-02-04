from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import re
import requests

app = FastAPI(title="Ownerclan Similar Finder API", version="1.0")


class SearchReq(BaseModel):
    url: str
    top_pages: int = 20
    mode: str = "main_only"
    seed_search_urls: Optional[List[str]] = []
    phash_threshold: int = Field(default=10, ge=0, le=32)
    max_candidates: int = Field(default=80, ge=10, le=500)
    only_first_image: bool = Field(default=False)


def extract_selfcode(product_url: str) -> str:
    m = re.search(r"selfcode=([A-Za-z0-9]+)", product_url)
    return m.group(1) if m else ""


def collect_candidate_urls_by_selfcode(selfcode: str, top_pages: int, max_candidates: int) -> List[str]:
    if not selfcode:
        return []

    base = "https://www.ownerclan.com/V2/search/search.php"
    found: List[str] = []
    seen = set()

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    }

    page_params = ["page", "p", "pg"]

    for p in range(1, max(1, top_pages) + 1):
        for pp in page_params:
            try:
                r = requests.get(
                    base,
                    params={"topSearchKeyword": selfcode, pp: p},
                    headers=headers,
                    timeout=10,
                )
                if r.status_code != 200:
                    continue

                html = r.text
                links = re.findall(r"/V2/product/view\.php\?selfcode=([A-Za-z0-9]+)", html)
                for code in links:
                    full = f"https://www.ownerclan.com/V2/product/view.php?selfcode={code}"
                    if full not in seen:
                        seen.add(full)
                        found.append(full)
                        if len(found) >= max_candidates:
                            return found
            except Exception:
                continue

    return found


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h2>Ownerclan Similar Finder API</h2>
    <ul>
      <li><a href="/openapi.json">/openapi.json</a></li>
      <li><a href="/docs">/docs</a></li>
      <li><a href="/privacy">/privacy</a></li>
    </ul>
    """


@app.post("/search", operation_id="searchOwnerclanSimilarProducts")
def search(req: SearchReq):
    selfcode = extract_selfcode(req.url)
    candidates = collect_candidate_urls_by_selfcode(selfcode, req.top_pages, req.max_candidates)

    # main-detail은 지금은 안전하게 비활성 (502 방지)
    if req.mode == "main-detail":
        return JSONResponse(
            status_code=200,
            content={
                "query_url": req.url,
                "mode": "main-detail-disabled",
                "top_pages": req.top_pages,
                "selfcode": selfcode,
                "candidate_urls_collected": len(candidates),
                "results": candidates[:20],
                "status": "ok",
                "note": "main-detail is temporarily disabled to prevent 502. Use main_only first.",
            },
        )

    return {
        "query_url": req.url,
        "mode": req.mode,
        "top_pages": req.top_pages,
        "selfcode": selfcode,
        "candidate_urls_collected": len(candidates),
        "results": candidates,
        "status": "ok",
    }


@app.post("/export_csv", operation_id="exportOwnerclanCsv")
def export_csv(req: SearchReq):
    return {"ok": True, "download_url": "/download/result.csv", "status": "ok"}


@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    return """
    <h1>Privacy Policy</h1>
    <p>This service does not store personal data.</p>
    <p>We only process the provided URL to return candidate product links.</p>
    """
