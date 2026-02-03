import asyncio
import re
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional, Dict, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field

from PIL import Image
import imagehash


app = FastAPI(title="Ownerclan Similar Finder API", version="1.1")


# -----------------------------
# Request Schema
# -----------------------------
class SearchReq(BaseModel):
    url: str
    top_pages: int = 20
    mode: str = "main_only"  # "main_only" or "main+detail"
    seed_search_urls: Optional[List[str]] = []

    phash_threshold: int = Field(default=10, ge=0, le=32)
    max_candidates: int = Field(default=80, ge=10, le=500)
    only_first_image: bool = Field(default=False)  # 대표이미지 1장 우선모드


# -----------------------------
# Utilities
# -----------------------------
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
)

IMG_EXT_RE = re.compile(r"\.(jpg|jpeg|png|webp)(\?|$)", re.IGNORECASE)

# 상세페이지와 무관한 이미지 후보를 거르는 강한 필터(경험적으로 흔한 패턴)
BAD_PATH_HINTS = [
    "logo", "icon", "sprite", "banner", "bn_", "btn", "button",
    "header", "footer", "top", "bottom", "gnb", "lnb", "aside",
    "common", "layout", "nav", "menu", "arrow",
    "bg", "background", "line", "bullet",
]

def is_probably_bad_image(url: str) -> bool:
    u = url.lower()
    return any(h in u for h in BAD_PATH_HINTS)

def normalize_abs_url(base_url: str, src: str) -> str:
    return urljoin(base_url, src.strip())

def domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

@dataclass
class ImgInfo:
    url: str
    phash: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None


# 간단 메모리 캐시 (속도 2~3배 튜닝 핵심 1)
_PHASH_CACHE: Dict[str, Tuple[str, int, int]] = {}  # img_url -> (phash, w, h)

# -----------------------------
# Network / Image Hash
# -----------------------------
async def fetch_text(client: httpx.AsyncClient, url: str) -> str:
    r = await client.get(url, follow_redirects=True, timeout=20)
    r.raise_for_status()
    return r.text

async def fetch_bytes(client: httpx.AsyncClient, url: str) -> bytes:
    r = await client.get(url, follow_redirects=True, timeout=25)
    r.raise_for_status()
    return r.content

async def compute_phash(client: httpx.AsyncClient, img_url: str) -> Optional[Tuple[str, int, int]]:
    if img_url in _PHASH_CACHE:
        return _PHASH_CACHE[img_url]

    # 이미지 확장자 아닌 것(또는 너무 수상한 것)은 스킵
    if not IMG_EXT_RE.search(img_url):
        return None

    try:
        raw = await fetch_bytes(client, img_url)
        im = Image.open(BytesIO(raw)).convert("RGB")
        w, h = im.size

        # 너무 작은 이미지는 로고/아이콘일 확률이 높아서 제거 (필터 핵심)
        if w < 250 or h < 250:
            return None

        ph = str(imagehash.phash(im))
        _PHASH_CACHE[img_url] = (ph, w, h)
        return ph, w, h
    except Exception:
        return None

def phash_distance(h1: str, h2: str) -> int:
    # imagehash는 hex string 기반이라, 다시 객체로 변환해 distance
    return int(imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2))


# -----------------------------
# HTML Parsing: "상품 이미지"만 추출
# -----------------------------
def extract_candidate_images_from_html(page_url: str, html: str) -> List[str]:
    """
    목표:
    - 상세페이지에서 "상품이미지로 보이는 것"만 추출
    - 로고/배너/아이콘/푸터/공통 이미지 최대한 제거
    전략:
    1) img 태그 src/data-src 모두 수집
    2) 절대URL 변환 + 도메인 기준 정리
    3) 경로 힌트/작은 이미지 제외
    4) 중복 제거
    """
    soup = BeautifulSoup(html, "lxml")
    imgs = []

    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-original")
        if not src:
            continue

        absu = normalize_abs_url(page_url, src)

        # 상세페이지와 관계 없는 외부 CDN/광고 이미지가 많으면 여기서 도메인 필터도 가능
        # (너무 빡세면 상품이미지도 날릴 수 있어 우선은 약하게)
        if is_probably_bad_image(absu):
            continue

        imgs.append(absu)

    # 중복 제거(순서 유지)
    seen = set()
    out = []
    for u in imgs:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)

    return out


# -----------------------------
# Main Logic
# -----------------------------
async def get_main_product_images(client: httpx.AsyncClient, product_url: str, only_first: bool) -> List[ImgInfo]:
    html = await fetch_text(client, product_url)
    urls = extract_candidate_images_from_html(product_url, html)

    # 대표이미지 1장 우선 모드
    if only_first and urls:
        urls = urls[:1]

    # phash 병렬 계산 (속도 2~3배 튜닝 핵심 2: 동시 처리)
    tasks = [compute_phash(client, u) for u in urls]
    results = await asyncio.gather(*tasks)

    img_infos: List[ImgInfo] = []
    for u, r in zip(urls, results):
        if not r:
            continue
        ph, w, h = r
        img_infos.append(ImgInfo(url=u, phash=ph, width=w, height=h))

    return img_infos


async def get_candidate_product_pages_from_seed(client: httpx.AsyncClient, seed_urls: List[str], limit: int) -> List[str]:
    """
    seed_search_urls 는 사용자가 직접 넣어주는 '검색결과 페이지 URL' 혹은 '카테고리/리스트 페이지 URL'
    여기서 상세상품 링크들을 추출합니다.
    """
    product_links: List[str] = []
    seen = set()

    for s_url in seed_urls:
        try:
            html = await fetch_text(client, s_url)
        except Exception:
            continue

        soup = BeautifulSoup(html, "lxml")
        # ownerclan은 /V2/product/view.php?selfcode=... 형태가 핵심
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "product/view.php" not in href:
                continue
            absu = normalize_abs_url(s_url, href)
            if absu in seen:
                continue
            seen.add(absu)
            product_links.append(absu)
            if len(product_links) >= limit:
                return product_links

    return product_links


async def score_candidate_by_images(
    client: httpx.AsyncClient,
    query_imgs: List[ImgInfo],
    candidate_url: str,
    phash_threshold: int,
    mode: str
) -> Optional[Dict]:
    """
    후보 상품 상세페이지에서 이미지 추출 → query 이미지와 phash 비교 → 최소 거리(가까울수록 유사)
    """
    try:
        html = await fetch_text(client, candidate_url)
    except Exception:
        return None

    cand_img_urls = extract_candidate_images_from_html(candidate_url, html)

    # mode에 따라 이미지 수를 제한(속도 튜닝)
    # - main_only: 앞에서 1~3장 정도만 빠르게 검사
    # - main+detail: 더 많이 검사
    if mode == "main_only":
        cand_img_urls = cand_img_urls[:3]
    else:
        cand_img_urls = cand_img_urls[:10]

    # 후보 이미지 phash 계산
    tasks = [compute_phash(client, u) for u in cand_img_urls]
    ph_results = await asyncio.gather(*tasks)

    cand_hashes = []
    for u, r in zip(cand_img_urls, ph_results):
        if not r:
            continue
        ph, w, h = r
        cand_hashes.append((u, ph))

    if not cand_hashes or not query_imgs:
        return None

    # query_imgs vs cand_hashes 최소 거리 계산
    best = None  # (dist, q_img_url, c_img_url)
    for qi in query_imgs:
        for cu, ch in cand_hashes:
            dist = phash_distance(qi.phash, ch)
            if best is None or dist < best[0]:
                best = (dist, qi.url, cu)

    if best is None:
        return None

    dist, qimg, cimg = best
    # threshold 이내만 유사로 판단
    if dist > phash_threshold:
        return None

    # 가격/배송비/총액은 "일반가 기준"으로 잡아야 하므로
    # 실제 파싱은 사이트 구조 확정 후 구현(지금은 빈 값으로 두고, GPT 출력에서 정렬은 total_price가 있으면 적용)
    return {
        "product_url": candidate_url,
        "match_distance": dist,
        "match_query_image": qimg,
        "match_candidate_image": cimg,
        "price": None,
        "shipping": None,
        "total_price": None,
    }


@app.post("/search")
async def search(req: SearchReq):
    """
    동작:
    1) 입력 상품 URL에서 '상품 이미지'만 필터링 추출 (로고/아이콘/배너 제외)
    2) 대표이미지 1장 우선 모드(옵션)
    3) seed_search_urls 로 준 검색/리스트 페이지에서 후보 상품 링크 추출
    4) 후보 상품 이미지와 phash 유사도 비교 → threshold 이내만 결과로 반환
    """
    async with httpx.AsyncClient(headers={"User-Agent": UA}) as client:
        query_imgs = await get_main_product_images(client, req.url, req.only_first_image)

        # seed가 없으면 유사상품 탐색이 불가능 → 이미지 추출까지만 확인해주고 종료
        if not req.seed_search_urls:
            return JSONResponse({
                "ok": True,
                "status": "connected",
                "query_url": req.url,
                "mode": req.mode,
                "only_first_image": req.only_first_image,
                "extracted_query_images": [i.url for i in query_imgs],
                "hint": "seed_search_urls(검색결과/리스트 페이지 URL)을 넣으면 후보 상품을 찾아 유사도 비교를 수행합니다."
            })

        candidate_pages = await get_candidate_product_pages_from_seed(
            client,
            req.seed_search_urls,
            limit=req.max_candidates
        )

        # 후보를 병렬로 점수화(속도 튜닝 핵심 3)
        tasks = [
            score_candidate_by_images(client, query_imgs, cu, req.phash_threshold, req.mode)
            for cu in candidate_pages
        ]
        scored = await asyncio.gather(*tasks)
        results = [r for r in scored if r]

        # 정렬 규칙:
        # 1) total_price(가격+배송비)가 있으면 total_price 오름차순
        # 2) 없으면 match_distance(유사도) 오름차순
        def sort_key(x):
            tp = x.get("total_price")
            if tp is None:
                return (10**9, x.get("match_distance", 999))
            return (tp, x.get("match_distance", 999))

        results.sort(key=sort_key)

        return {
            "ok": True,
            "status": "ok",
            "query_url": req.url,
            "mode": req.mode,
            "only_first_image": req.only_first_image,
            "extracted_query_images": [i.url for i in query_imgs],
            "candidates_checked": len(candidate_pages),
            "results_count": len(results),
            "results": results[: req.max_candidates],
        }


@app.post("/export_csv")
async def export_csv(req: SearchReq):
    # 현재는 download_url만 제공 (다음 단계에서 results를 서버 메모리/스토리지에 저장 후 CSV 생성)
    return {"ok": True, "download_url": "/download/result.csv", "status": "connected"}


@app.get("/download/result.csv", response_class=PlainTextResponse)
def download_csv():
    # 임시 CSV (다음 단계에서 실제 결과 저장/다운로드 구현)
    return "rank,product_url,match_distance,price,shipping,total_price\n"


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
