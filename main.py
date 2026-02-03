from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List, Optional
import re
import requests

app = FastAPI(title="Ownerclan Similar Finder API", version="1.0")


# -----------------------------
# 요청 스키마
# -----------------------------
class SearchReq(BaseModel):
    url: str
    top_pages: int = 20
    mode: str = "main_only"  # "main_only" | "main-detail" (현재는 안전하게 비활성)
    seed_search_urls: Optional[List[str]] = []

    # 아래 값들은 지금 단계에서는 서버 안정성용으로만 둠
    phash_threshold: int = Field(default=10, ge=0, le=32)
    max_candidates: int = Field(default=80, ge=10, le=500)
    only_first_image: bool = Field(default=False)


# -----------------------------
# 유틸: selfcode 추출
# -----------------------------
def extract_selfcode(product_url: str) -> str:
    m = re.search(r"selfcode=([A-Za-z0-9]+)", product_url)
    return m.group(1) if m else ""


# -----------------------------
# 핵심: 키워드(=selfcode)로 후보 URL 수집
#  - 지금은 "유사상품 찾기"의 1차 목표: 후보 URL이 0이 아니게 만들기
# -----------------------------
def collect_candidate_urls_by_selfcode(selfcode: str, top_pages: int, max_candidates: int) -> List[str]:
    """
    ownerclan 검색 페이지에서 selfcode가 포함된 결과 URL을 최대 max_candidates개 모음.
    - 사이트 구조가 바뀌면 여기만 손보면 됨
    """
    if not selfcode:
        return []

    base = "https://www.ownerclan.com/V2/search/search.php"
    found = []
    seen = set()

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    }

    # 페이지 파라미터는 사이트에 따라 다를 수 있어, 몇 가지 패턴을 같이 시도
    page_params = ["page", "p", "pg"]

    for p in range(1, max(1, top_pages) + 1):
        # 가능한 파라미터 패턴을 모두 시도
        tried = []
        for pp in page_params:
            params = {"topSearchKeyword": selfcode, pp: p}
            tried.append((base, params))

        for url, params in tried:
            try:
                r = requests.get(url, params=params, headers=headers, timeout=10)
                if r.status_code != 200:
                    continue
                html = r.text

                # 결과에서 product/view.php?selfcode=XXXX 형태의 링크 수집
                # (따옴표/슬래시 형태 다양해서 정규식으로 넓게)
                links = re.findall(r"/V2/product/view\.php\?selfcode=([A-Za-z0-9]+)", html)
                for code in links:
                    full = f"https://www.ownerclan.com/V2/product/view.php?selfcode={code}"
                    if full not in seen:
                        seen.add(full)
                        found.append(full)
                        if len(found) >= max_candidates:
                            return found
            except Exception:
                # 어떤 페이지에서 터져도 서버가 죽지 않게 무시
                continue

    return found


# -----------------------------
# API: /search
# -----------------------------
@app.post("/search", operation_id="searchOwnerclanSimilarProducts")
def search(req: SearchReq):
    # selfcode를 뽑아서 후보 URL 수집
    selfcode = extract_selfcode(req.url)
    candidates = collect_candidate_urls_by_selfcode(
        selfcode=selfcode,
        top_pages=req.top_pages,
        max_candidates=req.max_candidates
    )

    # main-detail은 일단 안전하게 막아둠 (502 방지)
    if req.mode == "main-detail":
        return JSONResponse(
            status_code=200,
            content={
                "query_url": req.url,
                "mode": "main-detail-disabled",
                "top_pages": req.top_pages,
                "selfcode": selfcode,
                "candidate_urls_collected": len(candidates),
                "results": candidates[:20],  # 미리보기로 20개만
                "status": "ok",
                "note": "main-detail is temporarily disabled to prevent 502. Use main_only first."
            }
        )

    # 기본(main_only): 후보 URL을 반환
    return {
        "query_url": req.url,
        "mode": req.mode,
        "top_pages": req.top_pages,
        "selfcode": selfcode,
        "candidate_urls_collected": len(candidates),
        "results": candidates,
        "status": "ok",
    }


# -----------------------------
# API: /export_csv (지금은 더미)
# -----------------------------
@app.post("/export_csv", operation_id="exportOwnerclanCsv")
def export_csv(req: SearchReq):
    return {
        "ok": True,
        "download_url": "/download/result.csv",
        "status": "ok"
    }


# -----------------------------
# Privacy Policy (GPT Actions용)
# -----------------------------
@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    return """
    <h1>Privacy Policy</h1>
    <p>This service does not store personal data.</p>
    <p>We only process the provided URL to return candidate product links.</p>
    """
