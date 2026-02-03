from typing import List, Optional
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Ownerclan Similar Finder API", version="1.0")


# -----------------------------
# Request / Response Schemas
# -----------------------------
class SearchReq(BaseModel):
    url: str
    top_pages: int = 20
    mode: str = "main_only"
    seed_search_urls: Optional[List[str]] = []

    phash_threshold: int = Field(default=10, ge=0, le=32)   # 낮을수록 엄격
    max_candidates: int = Field(default=80, ge=10, le=500)  # 후보 상품 최대
    only_first_image: bool = Field(default=False)           # 대표이미지 1장 우선 모드


# -----------------------------
# API Endpoints
# -----------------------------
@app.post("/search")
def search(req: SearchReq):
    """
    GPTs Action 연결 확인용(테스트용) 엔드포인트.
    - 지금 단계에서는 실제 크롤링/유사도 검색 로직 없이 '정상 연결'만 확인합니다.
    - 다음 단계에서 여기에 실제 로직(이미지 추출/필터/유사도/가격정렬/CSV)을 붙입니다.
    """
    return {
        "ok": True,
        "query_url": req.url,
        "mode": req.mode,
        "top_pages": req.top_pages,
        "seed_search_urls": req.seed_search_urls,
        "phash_threshold": req.phash_threshold,
        "max_candidates": req.max_candidates,
        "only_first_image": req.only_first_image,
        "results": [],  # 여기에 유사상품 결과가 들어가게 확장 예정
        "status": "connected",
    }


@app.post("/export_csv")
def export_csv(req: SearchReq):
    """
    CSV 다운로드 기능은 다음 단계에서 구현.
    지금은 액션 연결/응답 형식 확인용으로 download_url만 돌려줍니다.
    """
    return {
        "ok": True,
        "download_url": "/download/result.csv",
        "status": "connected",
    }


@app.get("/download/result.csv", response_class=PlainTextResponse)
def download_csv():
    """
    임시 CSV (테스트용)
    """
    csv_text = "rank,product_url,product_name,price,shipping,total_price,match_score\n"
    return csv_text


@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    return """
    <h1>Privacy Policy</h1>
    <p>No personal data is stored. This service only processes user-provided URLs for similarity search.</p>
    """


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h2>Ownerclan Similar Finder API</h2>
    <ul>
      <li>GET <a href="/privacy">/privacy</a></li>
      <li>GET <a href="/docs">/docs</a> (Swagger)</li>
      <li>GET <a href="/openapi.json">/openapi.json</a></li>
    </ul>
    """
