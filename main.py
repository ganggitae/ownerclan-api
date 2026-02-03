import io
import re
import csv
import time
import uuid
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse, quote

import httpx
from bs4 import BeautifulSoup

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from PIL import Image
import imagehash


# =========================
# Config
# =========================
BASE_DOMAIN = "www.ownerclan.com"
BASE_URL = "https://www.ownerclan.com/"
V2_SEARCH_PATH = "/V2/product/search.php"
PRODUCT_VIEW_RE = re.compile(r"/V2/product/view\.php\?selfcode=([A-Za-z0-9]+)")

# 이미지 제외 패턴(대충 '헤더/아이콘/배너/경고/알림'류)
EXCLUDE_IMG_SUBSTR = [
    "icon", "logo", "common", "header", "footer", "banner", "btn",
    "productAlert", "alert", "notice", "loading", "sprite",
]

# 너무 작은 이미지는 상세와 무관할 확률이 큼
MIN_IMG_EDGE = 220  # width/height 중 최소값

# 해시 유사도 기준(작을수록 더 비슷)
PHASH_THRESHOLD_MAIN = 10   # 대표이미지 비교 기준
PHASH_THRESHOLD_DETAIL = 12 # 상세이미지 비교 기준

# 상위 후보만 상세 비교(속도 최적화)
DETAIL_RECHECK_TOPK = 25

# 요청 결과 메모리 저장(간단 CSV용)
LAST_RESULTS: Dict[str, List[Dict]] = {}


# =========================
# API Models
# =========================
class SearchReq(BaseModel):
    url: str
    top_pages: int = 20
    mode: str = "main_only"
    # mode:
    # - "main_only": 대표이미지 1장 우선검색(빠름)
    # - "main+detail": 대표이미지로 1차 필터 후 상세이미지까지 2차 검증(추천)
    # - "detail_only": 상세 위주(느림)


app = FastAPI(title="Ownerclan Similar Finder API", version="1.0")


# =========================
# Helpers
# =========================
def _is_ownerclan_url(u: str) -> bool:
    try:
        p = urlparse(u)
        return p.scheme in ("http", "https") and p.netloc == BASE_DOMAIN
    except Exception:
        return False


def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _extract_selfcode(u: str) -> Optional[str]:
    m = PRODUCT_VIEW_RE.search(u)
    return m.group(1) if m else None


async def fetch_html(client: httpx.AsyncClient, url: str) -> str:
    r = await client.get(url, follow_redirects=True, timeout=20)
    r.raise_for_status()
    return r.text


def pick_keywords(product_name: str, limit: int = 3) -> List[str]:
    """
    아주 단순 키워드 추출:
    - 한글/영문/숫자 토큰 중 2글자 이상
    - 너무 흔한 단어는 제외(원하면 추가)
    """
    stop = {"무료", "배송", "정품", "국내", "세트", "할인", "특가", "상품", "구매", "판매"}
    tokens = re.findall(r"[A-Za-z0-9가-힣]{2,}", product_name)
    out = []
    for t in tokens:
        if t in stop:
            continue
        if len(out) >= limit:
            break
        out.append(t)
    return out[:limit]


def _looks_unrelated_img(src: str) -> bool:
    s = (src or "").lower()
    return any(x.lower() in s for x in EXCLUDE_IMG_SUBSTR)


async def fetch_image_bytes(client: httpx.AsyncClient, url: str) -> Optional[bytes]:
    try:
        r = await client.get(url, timeout=25, follow_redirects=True)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "image" not in ctype:
            return None
        return r.content
    except Exception:
        return None


def image_phash(img_bytes: bytes) -> Optional[imagehash.ImageHash]:
    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # 크기 기준으로 관련 없는 작은 이미지 제거
        if min(im.size[0], im.size[1]) < MIN_IMG_EDGE:
            return None
        return imagehash.phash(im)
    except Exception:
        return None


def hamming(a: imagehash.ImageHash, b: imagehash.ImageHash) -> int:
    return int(a - b)


def parse_product_page_for_images(html: str, page_url: str) -> Dict[str, List[str]]:
    """
    "상세페이지와 관계없는 이미지" 걸러내기 핵심:
    - og:image (대표이미지 후보)
    - 본문 상세영역 안의 img만(가능하면)
    - src가 이상하거나(아이콘/배너) 너무 흔한 패턴이면 제외
    """
    soup = BeautifulSoup(html, "lxml")

    main_imgs: List[str] = []
    detail_imgs: List[str] = []

    # 1) 대표: og:image
    og = soup.select_one("meta[property='og:image']")
    if og and og.get("content"):
        main_imgs.append(urljoin(page_url, og["content"]))

    # 2) 대표: 흔한 메인이미지 셀렉터 후보(사이트마다 다름 → 최대한 안전하게 여러 후보)
    for sel in [
        "img#mainImage", "img#bigimg", "div.product_view img", "div.prd_img img", "div.goods_img img"
    ]:
        for img in soup.select(sel):
            src = img.get("src") or img.get("data-src")
            if not src:
                continue
            u = urljoin(page_url, src)
            if _looks_unrelated_img(u):
                continue
            main_imgs.append(u)

    # 3) 상세: 상세설명 영역 후보
    for sel in [
        "div#goodsDetail img",
        "div#productDetail img",
        "div#prdDetail img",
        "div.detail img",
        "div.goods_detail img",
        "div.product_detail img",
        "div#contents img",
    ]:
        imgs = soup.select(sel)
        if imgs:
            for img in imgs:
                src = img.get("src") or img.get("data-src")
                if not src:
                    continue
                u = urljoin(page_url, src)
                if _looks_unrelated_img(u):
                    continue
                detail_imgs.append(u)
            break  # 첫 매칭 셀렉터만 사용(노이즈 줄이기)

    # 중복 제거(순서 유지)
    def uniq(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    return {
        "main": uniq(main_imgs)[:5],
        "detail": uniq(detail_imgs)[:50],
    }


def parse_product_name(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    # title 우선
    t = _clean_text(soup.title.get_text()) if soup.title else ""
    # 너무 길면 앞부분 사용
    return t[:80] if t else ""


def parse_price_and_shipping(html: str) -> Tuple[Optional[int], Optional[int]]:
    """
    '일반가'만 판단:
    - 사이트 구조가 확실치 않아서 '일반가' 라벨 근처 숫자를 최대한 안전하게 추출
    - 실패 시 None 반환
    """
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)

    def find_money_after(label: str) -> Optional[int]:
        m = re.search(label + r".{0,30}?([0-9][0-9,]{2,})\s*원", text)
        if not m:
            return None
        return int(m.group(1).replace(",", ""))

    # 일반가
    price = find_money_after("일반가")
    # 배송비
    ship = find_money_after("배송비")
    return price, ship


async def collect_candidate_product_urls(client: httpx.AsyncClient, keyword: str, top_pages: int) -> List[str]:
    """
    오너클랜 내부 검색 페이지에서 후보 상품 URL 수집
    - top_pages 만큼 '다음 페이지'를 어떻게 넘기는지 사이트마다 다르니,
      일단 단일 페이지에서 최대한 많이 모으는 방식 + 페이지 파라미터 추정(있으면)
    """
    out = []

    # 1) 기본 검색 URL
    q = quote(keyword)
    base_search_url = f"{BASE_URL.rstrip('/')}{V2_SEARCH_PATH}?topSearchKeywordInfo=&topSearchKeyword={q}&topSearchType=all"
    html = await fetch_html(client, base_search_url)
    soup = BeautifulSoup(html, "lxml")

    # 2) view.php?selfcode= 링크 모으기
    for a in soup.select("a[href]"):
        href = a.get("href") or ""
        if "view.php?selfcode=" in href:
            u = urljoin(base_search_url, href)
            if _is_ownerclan_url(u):
                out.append(u)

    # 3) (옵션) page 파라미터가 있는 경우를 대비한 추가 시도
    #    ※ 사이트가 page= 를 지원하지 않으면 그냥 무시됨.
    for page in range(2, max(2, top_pages + 1)):
        paged = base_search_url + f"&page={page}"
        try:
            html2 = await fetch_html(client, paged)
        except Exception:
            break
        soup2 = BeautifulSoup(html2, "lxml")
        found = 0
        for a in soup2.select("a[href]"):
            href = a.get("href") or ""
            if "view.php?selfcode=" in href:
                u = urljoin(paged, href)
                if _is_ownerclan_url(u):
                    out.append(u)
                    found += 1
        if found == 0:
            break

    # 중복 제거
    uniq = []
    seen = set()
    for u in out:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)
    return uniq


async def get_hashes_for_urls(client: httpx.AsyncClient, urls: List[str]) -> Dict[str, imagehash.ImageHash]:
    """
    대표이미지 1장만 다운로드해서 해시 생성
    """
    hashes: Dict[str, imagehash.ImageHash] = {}

    async def worker(u: str):
        try:
            html = await fetch_html(client, u)
            imgs = parse_product_page_for_images(html, u)
            main_list = imgs["main"]
            if not main_list:
                return
            b = await fetch_image_bytes(client, main_list[0])
            if not b:
                return
            h = image_phash(b)
            if h is None:
                return
            hashes[u] = h
        except Exception:
            return

    # 병렬(속도 2~3배 튜닝 핵심)
    # 너무 과하면 차단될 수 있으니 적당히 제한
    sem = httpx.Limits(max_connections=20, max_keepalive_connections=10)

    # 이미 client 자체가 limits를 가질 수 있으나, 안전하게 여기선 worker만 gather
    tasks = [worker(u) for u in urls]
    await asyncio_gather_limited(tasks, limit=20)
    return hashes


async def asyncio_gather_limited(coros, limit: int = 20):
    """
    asyncio 세마포어 기반 제한 gather
    """
    import asyncio
    sem = asyncio.Semaphore(limit)

    async def run(c):
        async with sem:
            return await c

    return await asyncio.gather(*[run(c) for c in coros], return_exceptions=True)


# =========================
# Core Search
# =========================
import asyncio

@app.post("/search")
async def search(req: SearchReq):
    if not _is_ownerclan_url(req.url):
        return {"status": "error", "message": "ownerclan.com URL만 허용됩니다.", "results": []}

    mode = req.mode.strip()
    if mode not in ("main_only", "main+detail", "detail_only"):
        mode = "main_only"

    request_id = str(uuid.uuid4())[:8]

    async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}, timeout=25) as client:
        # 1) 입력 상세페이지 읽기
        try:
            src_html = await fetch_html(client, req.url)
        except Exception as e:
            return {"status": "error", "message": f"입력 URL 로드 실패: {e}", "results": []}

        product_name = parse_product_name(src_html)
        imgs = parse_product_page_for_images(src_html, req.url)

        # 2) 입력 이미지 해시 만들기
        query_main_hashes: List[imagehash.ImageHash] = []
        query_detail_hashes: List[imagehash.ImageHash] = []

        # 대표 이미지(최우선)
        for u in imgs["main"][:1]:
            b = await fetch_image_bytes(client, u)
            if not b:
                continue
            h = image_phash(b)
            if h:
                query_main_hashes.append(h)

        # 상세 이미지(옵션)
        if mode in ("main+detail", "detail_only"):
            for u in imgs["detail"][:8]:
                b = await fetch_image_bytes(client, u)
                if not b:
                    continue
                h = image_phash(b)
                if h:
                    query_detail_hashes.append(h)

        if not query_main_hashes and mode != "detail_only":
            return {"status": "error", "message": "대표 이미지 해시 생성 실패(이미지 추출/다운로드 실패)", "results": []}
        if mode == "detail_only" and not query_detail_hashes:
            return {"status": "error", "message": "상세 이미지 해시 생성 실패", "results": []}

        # 3) 키워드 뽑기 → 후보 URL 수집(사이트 내부만)
        keywords = pick_keywords(product_name, limit=3)
        if not keywords:
            # 상품명이 없으면 selfcode 기반으로라도 진행(최소)
            sc = _extract_selfcode(req.url)
            keywords = [sc] if sc else []

        candidate_urls: List[str] = []
        for kw in keywords:
            urls = await collect_candidate_product_urls(client, kw, top_pages=req.top_pages)
            candidate_urls.extend(urls)

        # 중복 제거 + 자기 자신 제거
        seen = set()
        dedup = []
        for u in candidate_urls:
            if u == req.url:
                continue
            if u in seen:
                continue
            seen.add(u)
            dedup.append(u)
        candidate_urls = dedup

        # 후보가 너무 많으면 상한(속도/차단 방지)
        candidate_urls = candidate_urls[:300]

        # 4) 1차: 후보 대표이미지 해시만 만들어 빠르게 비교
        #    (여기가 속도 2~3배 핵심: 병렬 + 대표 1장만)
        #    ※ 간단히 구현: 아래에서 바로 병렬 생성/비교
        results = []

        async def score_candidate(u: str):
            try:
                html = await fetch_html(client, u)
                cand_imgs = parse_product_page_for_images(html, u)
                if not cand_imgs["main"]:
                    return None

                b = await fetch_image_bytes(client, cand_imgs["main"][0])
                if not b:
                    return None
                cand_main_h = image_phash(b)
                if cand_main_h is None:
                    return None

                # 대표 해시 vs 대표 해시
                if mode != "detail_only":
                    best_main = min(hamming(qh, cand_main_h) for qh in query_main_hashes)
                    if best_main > PHASH_THRESHOLD_MAIN:
                        return None
                else:
                    best_main = 999

                price, ship = parse_price_and_shipping(html)
                total = None
                if price is not None:
                    total = price + (ship or 0)

                return {
                    "url": u,
                    "product_name": parse_product_name(html),
                    "normal_price": price,
                    "shipping_fee": ship,
                    "total_price": total,
                    "score_main": best_main,
                    "score_detail": None,
                }
            except Exception:
                return None

        cand_scored = []
        await asyncio_gather_limited([asyncio.to_thread(lambda: None) for _ in []], limit=1)  # noop

        # 병렬 처리
        scored_list = await asyncio_gather_limited([score_candidate(u) for u in candidate_urls], limit=20)
        for item in scored_list:
            if isinstance(item, dict):
                cand_scored.append(item)

        # 대표 점수 낮은 순 정렬
        cand_scored.sort(key=lambda x: (x["score_main"] if x["score_main"] is not None else 999, x["total_price"] or 10**18))

        # 5) 2차(선택): 상위 후보만 상세 이미지로 재검증
        if mode == "main+detail" and query_detail_hashes:
            topk = cand_scored[:DETAIL_RECHECK_TOPK]

            async def detail_recheck(item: Dict):
                try:
                    html = await fetch_html(client, item["url"])
                    cand_imgs = parse_product_page_for_images(html, item["url"])
                    # 후보 상세 이미지 일부만
                    durls = cand_imgs["detail"][:10]
                    cand_detail_hashes = []
                    for du in durls:
                        b = await fetch_image_bytes(client, du)
                        if not b:
                            continue
                        h = image_phash(b)
                        if h:
                            cand_detail_hashes.append(h)

                    if not cand_detail_hashes:
                        item["score_detail"] = None
                        return item

                    best_detail = min(hamming(qh, ch) for qh in query_detail_hashes for ch in cand_detail_hashes)
                    item["score_detail"] = best_detail

                    # 상세 기준으로 컷
                    if best_detail > PHASH_THRESHOLD_DETAIL:
                        return None
                    return item
                except Exception:
                    return None

            rechecked = await asyncio_gather_limited([detail_recheck(x) for x in topk], limit=10)
            kept = [x for x in rechecked if isinstance(x, dict)]
            # 상세점수 우선 정렬
            kept.sort(key=lambda x: (x["score_detail"] if x["score_detail"] is not None else 999, x["total_price"] or 10**18))
            results = kept
        else:
            results = cand_scored

        # 6) 가격 정렬 규칙:
        # - "일반가"만 사용(normal_price)
        # - 배송비 있으면 total_price로 정렬
        # - 없으면 normal_price로 정렬
        def sort_key(x):
            # total이 있으면 total이 우선, 없으면 normal
            t = x.get("total_price")
            p = x.get("normal_price")
            return (t if t is not None else 10**18, p if p is not None else 10**18, x.get("score_main") or 999)

        results.sort(key=sort_key)

        # 결과 상한
        results = results[:50]

        # CSV용 저장
        LAST_RESULTS[request_id] = results

        return {
            "status": "ok",
            "request_id": request_id,
            "query_url": req.url,
            "query_product_name": product_name,
            "keywords_used": keywords,
            "mode": mode,
            "candidate_count": len(candidate_urls),
            "result_count": len(results),
            "results": results,
            "csv_download_url": f"/download/result.csv?request_id={request_id}",
        }


@app.get("/download/result.csv")
async def download_csv(request_id: str):
    rows = LAST_RESULTS.get(request_id, [])
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "url", "product_name", "normal_price", "shipping_fee", "total_price", "score_main", "score_detail"
    ])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

    mem = io.BytesIO(output.getvalue().encode("utf-8-sig"))
    return StreamingResponse(
        mem,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="ownerclan_results_{request_id}.csv"'}
    )


@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    return """
    <h1>Privacy Policy</h1>
    <p>This API does not store personal data. It only processes provided URLs to return similarity results.</p>
    """


@app.get("/")
def root():
    return {"ok": True, "message": "Ownerclan Similar Finder API is running. Use /search."}

from fastapi.responses import JSONResponse

@app.get("/")
def root():
    return JSONResponse({"ok": True, "message": "Ownerclan Similar Finder API is running. Use /search."})
