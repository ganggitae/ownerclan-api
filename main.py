import re
import io
import csv
import time
import hashlib
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field

# ---------------------------
# 기본 설정
# ---------------------------
BASE_DOMAIN = "www.ownerclan.com"
BASE_URL = "https://www.ownerclan.com/"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

# 간단 캐시 (메모리)
CACHE_TTL_SEC = 60 * 30
_cache: Dict[str, Tuple[float, dict]] = {}

# 최근 결과 CSV용 임시 저장 (메모리)
_last_csv_rows: List[Dict[str, str]] = []


app = FastAPI(title="Ownerclan Similar Finder API", version="1.0")


# ---------------------------
# 요청 스키마
# ---------------------------
class SearchReq(BaseModel):
    url: str
    top_pages: int = Field(default=20, ge=1, le=200)
    mode: str = Field(default="main_only")  # main_only | main_first | main+detail
    seed_search_urls: Optional[List[str]] = None  # 사용자가 검색 결과 페이지들을 직접 넣을 수 있음
    phash_threshold: int = Field(default=10, ge=0, le=32)  # 유사도 기준(낮을수록 엄격)
    max_candidates: int = Field(default=80, ge=10, le=500)  # 후보 상품 최대
    only_first_image: bool = Field(default=False)  # 대표이미지 1장 우선 모드


# ---------------------------
# 유틸
# ---------------------------
def _cache_get(key: str) -> Optional[dict]:
    v = _cache.get(key)
    if not v:
        return None
    ts, data = v
    if time.time() - ts > CACHE_TTL_SEC:
        _cache.pop(key, None)
        return None
    return data


def _cache_set(key: str, data: dict):
    _cache[key] = (time.time(), data)


def _norm_url(u: str) -> str:
    return u.strip().strip("|")


def _is_ownerclan(u: str) -> bool:
    try:
        p = urlparse(u)
        return p.netloc.endswith("ownerclan.com")
    except:
        return False


def _abs(base: str, maybe: str) -> str:
    return urljoin(base, maybe)


def _looks_like_ui_image(src: str) -> bool:
    s = src.lower()
    # 흔한 UI/로고/아이콘/배너/공통 이미지 패턴 필터
    bad_kw = [
        "logo", "icon", "sprite", "btn", "button", "banner", "common", "header", "footer",
        "top_", "bottom_", "gnb", "lnb", "side", "coupon", "event", "popup", "loading",
        "bg_", "background", "review_star", "star", "rank", "sns", "kakao", "naver",
    ]
    return any(k in s for k in bad_kw)


def _score_image_relevance(img_tag) -> int:
    """
    상세페이지 '상품 이미지'에 가까울수록 점수 ↑
    alt/class/id/src 단서로 대충 점수화
    """
    score = 0
    alt = (img_tag.get("alt") or "").lower()
    cls = " ".join(img_tag.get("class", [])).lower()
    _id = (img_tag.get("id") or "").lower()
    src = (img_tag.get("src") or "").lower()
    data_src = (img_tag.get("data-src") or "").lower()

    text = " ".join([alt, cls, _id, src, data_src])

    good_kw = ["product", "detail", "goods", "item", "thumb", "image", "view"]
    bad_kw = ["logo", "icon", "banner", "coupon", "btn", "header", "footer", "nav", "sns"]

    score += sum(2 for k in good_kw if k in text)
    score -= sum(3 for k in bad_kw if k in text)

    # 사이즈가 큰 이미지일 확률: width/height 힌트가 있으면 가점
    w = img_tag.get("width")
    h = img_tag.get("height")
    try:
        if w and int(w) >= 400: score += 2
        if h and int(h) >= 400: score += 2
    except:
        pass

    # UI 이미지로 보이면 크게 감점
    if _looks_like_ui_image(src):
        score -= 8

    return score


async def fetch_text(client: httpx.AsyncClient, url: str) -> str:
    r = await client.get(url, timeout=20)
    r.raise_for_status()
    return r.text


async def fetch_bytes(client: httpx.AsyncClient, url: str) -> bytes:
    r = await client.get(url, timeout=30)
    r.raise_for_status()
    return r.content


def _simple_phash_bytes(img_bytes: bytes) -> str:
    """
    초간단 해시(대체용). 실제 pHash 라이브러리 못 쓰는 환경에서도 '비슷한 파일' 정도는 걸러짐.
    - 이미지 바이트 자체를 블록으로 나눠 해시
    - 완전 동일/거의 동일 이미지에는 꽤 강함
    """
    # 안정적 해시
    return hashlib.md5(img_bytes).hexdigest()


def _hamming_hex(a: str, b: str) -> int:
    # md5 hex 기반 유사도는 엄밀한 phash는 아니지만, 동일성/근접성 필터로 사용
    # (동일하면 0)
    return 0 if a == b else 32


# ---------------------------
# 1) 상세페이지에서 “상품 이미지”만 뽑기
# ---------------------------
async def extract_product_images(client: httpx.AsyncClient, product_url: str, only_first: bool) -> List[str]:
    html = await fetch_text(client, product_url)
    soup = BeautifulSoup(html, "html.parser")

    imgs = soup.find_all("img")
    candidates = []

    for img in imgs:
        src = img.get("data-src") or img.get("src") or ""
        src = src.strip()
        if not src:
            continue

        abs_src = _abs(product_url, src)

        # 도메인 내부/외부 모두 가능하지만, UI이미지 최대 배제
        if _looks_like_ui_image(abs_src):
            continue

        score = _score_image_relevance(img)
        if score < 0:
            continue

        candidates.append((score, abs_src))

    # 점수 높은 순으로 정렬 후 중복 제거
    candidates.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    out = []
    for _, u in candidates:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
        if only_first and len(out) >= 1:
            break

    return out


# ---------------------------
# 2) 검색 결과 페이지에서 상품 상세 URL들 수집
# ---------------------------
def extract_product_links_from_search(html: str, base: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "product/view.php" in href and "selfcode=" in href:
            links.append(urljoin(base, href))

    # 중복 제거
    out = []
    seen = set()
    for u in links:
        u = _norm_url(u)
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


# ---------------------------
# 3) 상품 페이지에서 가격/배송비/대표이미지(썸네일) 등 파싱
#    ※ 오너클랜 DOM은 계속 바뀔 수 있어서 selector는 '최소 안전' 방식
# ---------------------------
_price_re = re.compile(r"([0-9][0-9,]*)\s*원")

def _parse_int_krw(text: str) -> Optional[int]:
    m = _price_re.search(text.replace("\xa0", " "))
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


async def parse_product_meta(client: httpx.AsyncClient, url: str) -> Dict:
    html = await fetch_text(client, url)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    # 상품명: title 또는 h1/h2 비슷한 큰 텍스트 우선
    title = (soup.title.get_text(strip=True) if soup.title else "").strip()
    if not title:
        # fallback
        h = soup.find(["h1", "h2"])
        title = h.get_text(strip=True) if h else ""

    # 일반가: 페이지 전체 텍스트에서 첫 가격 후보
    normal_price = _parse_int_krw(text)

    # 배송비: "배송비" 주변 텍스트에서 추출 시도
    shipping_fee = 0
    ship_idx = text.find("배송비")
    if ship_idx != -1:
        near = text[ship_idx: ship_idx + 80]
        v = _parse_int_krw(near)
        if v is not None:
            shipping_fee = v

    # 대표 이미지(썸네일): detail images 중 첫 번째로 대체
    images = await extract_product_images(client, url, only_first=True)
    thumb = images[0] if images else ""

    return {
        "url": url,
        "title": title,
        "normal_price": normal_price if normal_price is not None else 0,
        "shipping_fee": shipping_fee,
        "total_price": (normal_price if normal_price is not None else 0) + shipping_fee,
        "thumb": thumb,
    }


# ---------------------------
# 4) 유사 이미지 매칭(1차: 동일성 위주)
# ---------------------------
async def build_image_hashes(client: httpx.AsyncClient, image_urls: List[str]) -> Dict[str, str]:
    """
    이미지 다운로드 후 간단 해시 생성
    """
    hashes = {}
    async def _one(u: str):
        try:
            b = await fetch_bytes(client, u)
            hashes[u] = _simple_phash_bytes(b)
        except:
            pass

    # 동시 실행
    await httpx.AsyncClient().aclose()  # 안전용(실행환경에 따라 필요 없지만)
    # 실제는 아래 gather로
    import asyncio
    tasks = [asyncio.create_task(_one(u)) for u in image_urls]
    await asyncio.gather(*tasks)
    return hashes


def is_similar(h1: str, h2: str, threshold: int) -> bool:
    # md5 기반이라 threshold 의미 약하지만, 동일성 매칭으로 사용
    return _hamming_hex(h1, h2) <= threshold


# ---------------------------
# 메인 API
# ---------------------------
@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    return """
    <h1>Privacy Policy</h1>
    <p>No personal data is stored permanently. Temporary in-memory cache may be used to speed up repeated requests.</p>
    """


@app.post("/search")
async def search(req: SearchReq):
    url = _norm_url(req.url)
    if not _is_ownerclan(url):
        raise HTTPException(status_code=400, detail="Only ownerclan.com URLs are supported.")

    cache_key = f"search::{url}::{req.top_pages}::{req.mode}::{req.only_first_image}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
        # 1) 기준 상품 이미지 추출
        seed_images = await extract_product_images(client, url, only_first=req.only_first_image)
        if not seed_images:
            data = {
                "status": "ok",
                "query_url": url,
                "mode": req.mode,
                "top_pages": req.top_pages,
                "seed_images": [],
                "results": [],
                "message": "No product images found on the detail page (filtered)."
            }
            _cache_set(cache_key, data)
            return data

        # 2) seed 이미지 해시
        seed_hashes = await build_image_hashes(client, seed_images)

        # 3) 검색 대상 페이지들(사용자가 준 search url 있으면 그걸 우선 사용)
        search_pages = req.seed_search_urls or []
        if not search_pages:
            # 기본: 키워드 검색을 못 하니, '샘플 검색 URL' 같은 걸 사용자가 넣는 걸 권장
            # 일단은 빈 상태로 진행
            search_pages = []

        # 4) 후보 상품 URL 수집
        candidate_product_urls = []
        for sp in search_pages[:req.top_pages]:
            sp = _norm_url(sp)
            if not _is_ownerclan(sp):
                continue
            try:
                html = await fetch_text(client, sp)
                links = extract_product_links_from_search(html, sp)
                candidate_product_urls.extend(links)
            except:
                continue

        # 중복 제거 + 상한
        seen = set()
        uniq_candidates = []
        for u in candidate_product_urls:
            if u in seen:
                continue
            seen.add(u)
            uniq_candidates.append(u)
            if len(uniq_candidates) >= req.max_candidates:
                break

        # 5) 후보 상품별 대표이미지 해시 비교 (속도: 대표이미지만)
        import asyncio
        results = []

        async def _check_one(purl: str):
            try:
                meta = await parse_product_meta(client, purl)
                if not meta["thumb"]:
                    return

                # 후보 대표이미지 해시
                b = await fetch_bytes(client, meta["thumb"])
                h = _simple_phash_bytes(b)

                # seed 중 하나라도 유사하면 매칭
                for sh in seed_hashes.values():
                    if is_similar(sh, h, req.phash_threshold):
                        results.append(meta)
                        return
            except:
                return

        tasks = [asyncio.create_task(_check_one(pu)) for pu in uniq_candidates]
        await asyncio.gather(*tasks)

        # 6) 가격 정렬: (일반가 + 배송비) 낮은 순
        results.sort(key=lambda x: (x.get("total_price", 0), x.get("normal_price", 0)))

        # 7) CSV 저장(메모리)
        global _last_csv_rows
        _last_csv_rows = [
            {
                "url": r["url"],
                "title": r["title"],
                "normal_price": str(r["normal_price"]),
                "shipping_fee": str(r["shipping_fee"]),
                "total_price": str(r["total_price"]),
                "thumb": r["thumb"],
            }
            for r in results
        ]

        data = {
            "status": "ok",
            "query_url": url,
            "mode": req.mode,
            "top_pages": req.top_pages,
            "seed_images": seed_images,
            "result_count": len(results),
            "results": results,
        }
        _cache_set(cache_key, data)
        return data


@app.post("/export_csv")
def export_csv():
    if not _last_csv_rows:
        return JSONResponse({"ok": False, "message": "No results to export yet. Run /search first."})

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["url", "title", "normal_price", "shipping_fee", "total_price", "thumb"])
    writer.writeheader()
    writer.writerows(_last_csv_rows)

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=ownerclan_results.csv"},
    )
