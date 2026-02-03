import os
import re
import csv
import io
import time
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlparse, parse_qs, quote_plus

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse, Response
from pydantic import BaseModel, Field
from PIL import Image
import imagehash

app = FastAPI(title="Ownerclan Similar Finder API", version="1.0")

# -----------------------------
# Config
# -----------------------------
DEFAULT_TIMEOUT = 12
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)

# Railway/Render 등에서 프록시 뒤일 때도 잘 돌게끔
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})

# 마지막 CSV를 /download/result.csv 로 내려주기 위한 메모리 저장
LAST_CSV_BYTES: bytes = b""
LAST_CSV_TS: float = 0.0


# -----------------------------
# Request schema
# -----------------------------
class SearchReq(BaseModel):
    url: str
    top_pages: int = 20
    mode: str = "main_only"  # "main_only" | "main+detail"
    seed_search_urls: Optional[List[str]] = []
    phash_threshold: int = Field(default=10, ge=0, le=32)   # 낮을수록 엄격
    max_candidates: int = Field(default=80, ge=10, le=500)  # 후보 최대
    only_first_image: bool = False  # True면 후보/시드 모두 첫 이미지 1장만 비교


# -----------------------------
# Utilities
# -----------------------------
def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s


def _is_ownerclan_product_url(u: str) -> bool:
    return "ownerclan.com" in u and "/V2/product/view.php" in u and "selfcode=" in u


def _extract_selfcode(u: str) -> Optional[str]:
    try:
        q = parse_qs(urlparse(u).query)
        sc = q.get("selfcode", [None])[0]
        return sc
    except Exception:
        return None


def _absolute_url(base: str, href: str) -> str:
    if not href:
        return ""
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("//"):
        return "https:" + href
    # relative
    p = urlparse(base)
    return f"{p.scheme}://{p.netloc}{href if href.startswith('/') else '/' + href}"


def fetch_html(url: str) -> str:
    r = SESSION.get(url, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.text


def try_fetch_json(url: str) -> Optional[Dict[str, Any]]:
    try:
        r = SESSION.get(url, timeout=DEFAULT_TIMEOUT)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def parse_product_page(url: str) -> Dict[str, Any]:
    """
    Returns:
      {
        "title": str,
        "images": [url1, url2, ...],
        "selfcode": str|None
      }
    """
    html = fetch_html(url)
    soup = BeautifulSoup(html, "lxml")

    # title 우선순위: og:title -> title tag -> h1
    title = ""
    og_title = soup.select_one('meta[property="og:title"]')
    if og_title and og_title.get("content"):
        title = _clean_text(og_title["content"])

    if not title:
        t = soup.select_one("title")
        if t:
            title = _clean_text(t.get_text())

    if not title:
        h1 = soup.select_one("h1")
        if h1:
            title = _clean_text(h1.get_text())

    # 이미지 수집 우선순위:
    # 1) og:image
    # 2) img 태그 중 상품 영역에서 많이 나오는 것들
    images = []

    og_img = soup.select_one('meta[property="og:image"]')
    if og_img and og_img.get("content"):
        images.append(_absolute_url(url, og_img["content"]))

    # img 태그에서 후보 찾기 (너무 잡다한 아이콘 제외)
    for img in soup.select("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-original")
        if not src:
            continue
        absu = _absolute_url(url, src)
        if "favicon" in absu.lower():
            continue
        # 너무 작은 아이콘/스프라이트 추정 제외
        if any(x in absu.lower() for x in ["icon", "sprite", "loading"]):
            continue
        # 오너클랜 이미지 CDN/경로에 자주 들어가는 키워드 가중
        if "image" in absu.lower() or "img" in absu.lower() or "product" in absu.lower():
            images.append(absu)

    # 중복 제거(순서 유지)
    dedup = []
    seen = set()
    for im in images:
        if im and im not in seen:
            seen.add(im)
            dedup.append(im)

    return {
        "title": title,
        "images": dedup,
        "selfcode": _extract_selfcode(url),
    }


def build_search_urls(keyword: str, page: int) -> List[str]:
    """
    오너클랜이 UI/버전에 따라 검색 URL이 달라질 수 있어서 여러 패턴을 동시에 시도한다.
    """
    kw = quote_plus(keyword)

    # 패턴1: /V2/search/search.php 형태(추정)
    u1 = f"https://www.ownerclan.com/V2/search/search.php?topSearchKeyword={kw}&topSearchType=all&page={page}"

    # 패턴2: /V2/search/ + query 형태(추정)
    u2 = f"https://www.ownerclan.com/V2/search/?topSearchKeyword={kw}&topSearchType=all&page={page}"

    # 패턴3: /V2/product/search.php 형태(추정)
    u3 = f"https://www.ownerclan.com/V2/product/search.php?topSearchKeyword={kw}&topSearchType=all&page={page}"

    return [u1, u2, u3]


def parse_product_links_from_search(html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if "/V2/product/view.php" in href and "selfcode=" in href:
            links.append(_absolute_url("https://www.ownerclan.com", href))

    # 중복 제거
    out = []
    seen = set()
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def download_image_bytes(url: str) -> Optional[bytes]:
    try:
        r = SESSION.get(url, timeout=DEFAULT_TIMEOUT)
        if r.status_code != 200:
            return None
        return r.content
    except Exception:
        return None


def phash_from_bytes(b: bytes) -> Optional[imagehash.ImageHash]:
    try:
        im = Image.open(io.BytesIO(b))
        # 투명/팔레트 경고 방지용 변환
        im = im.convert("RGBA")
        return imagehash.phash(im)
    except Exception:
        return None


def collect_seed_images(seed_url: str, mode: str, only_first: bool) -> Tuple[str, List[str]]:
    info = parse_product_page(seed_url)
    title = info["title"] or ""
    imgs = info["images"] or []

    if mode == "main_only":
        imgs = imgs[:1]
    else:
        # main+detail 모드일 때도 너무 많으면 과부하 나므로 상한
        imgs = imgs[:12]

    if only_first:
        imgs = imgs[:1]

    return title, imgs


def collect_candidate_urls_by_keyword(keyword: str, top_pages: int, max_candidates: int) -> List[str]:
    """
    여러 검색 URL 패턴을 시도하면서 후보 URL을 모은다.
    """
    candidates = []
    seen = set()

    for page in range(1, max(1, top_pages) + 1):
        urls = build_search_urls(keyword, page)
        page_links: List[str] = []

        for su in urls:
            try:
                html = fetch_html(su)
                links = parse_product_links_from_search(html)
                if links:
                    page_links.extend(links)
            except Exception:
                continue

        for u in page_links:
            if u not in seen:
                seen.add(u)
                candidates.append(u)
                if len(candidates) >= max_candidates:
                    return candidates

        # 너무 빠르게 긁으면 차단될 수 있어 약간 쉬기
        time.sleep(0.15)

    return candidates


def compare_and_rank(
    seed_imgs: List[str],
    candidate_urls: List[str],
    phash_threshold: int,
    only_first: bool,
    mode: str,
    seed_url: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    seed 이미지들과 candidate 상품 이미지들을 pHash로 비교해 결과를 만든다.
    """
    # seed phash 준비
    seed_hashes = []
    seed_hash_map = {}  # seed_img_url -> hash
    for s in seed_imgs:
        b = download_image_bytes(s)
        if not b:
            continue
        h = phash_from_bytes(b)
        if h is None:
            continue
        seed_hashes.append((s, h))
        seed_hash_map[s] = h

    results = []
    scanned = 0

    for cu in candidate_urls:
        if cu == seed_url:
            continue
        # selfcode 같은 상품 제외
        if _extract_selfcode(cu) == _extract_selfcode(seed_url):
            continue

        scanned += 1
        try:
            pinfo = parse_product_page(cu)
            cimgs = pinfo["images"] or []
            if mode == "main_only":
                cimgs = cimgs[:1]
            else:
                cimgs = cimgs[:10]
            if only_first:
                cimgs = cimgs[:1]

            best = None  # (dist, seed_img, cand_img)
            for cimg in cimgs:
                cb = download_image_bytes(cimg)
                if not cb:
                    continue
                ch = phash_from_bytes(cb)
                if ch is None:
                    continue

                for s_img, s_h in seed_hashes:
                    dist = (s_h - ch)
                    if best is None or dist < best[0]:
                        best = (dist, s_img, cimg)

            if best and best[0] <= phash_threshold:
                results.append(
                    {
                        "candidate_url": cu,
                        "candidate_title": pinfo.get("title", ""),
                        "best_distance": int(best[0]),
                        "matched_seed_image": best[1],
                        "matched_candidate_image": best[2],
                    }
                )
        except Exception:
            continue

        # 과부하 방지
        if scanned % 10 == 0:
            time.sleep(0.2)

    # best_distance 오름차순 정렬
    results.sort(key=lambda x: x.get("best_distance", 999))

    debug = {
        "seed_image_count": len(seed_hashes),
        "candidate_count_scanned": scanned,
        "phash_threshold": phash_threshold,
    }
    return results, debug


def make_csv_bytes(seed_url: str, seed_title: str, results: List[Dict[str, Any]]) -> bytes:
    output = io.StringIO()
    w = csv.writer(output)
    w.writerow(["seed_url", "seed_title", "candidate_url", "candidate_title", "best_distance", "matched_seed_image", "matched_candidate_image"])
    for r in results:
        w.writerow([
            seed_url,
            seed_title,
            r.get("candidate_url", ""),
            r.get("candidate_title", ""),
            r.get("best_distance", ""),
            r.get("matched_seed_image", ""),
            r.get("matched_candidate_image", ""),
        ])
    return output.getvalue().encode("utf-8-sig")


# -----------------------------
# API endpoints
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h2>Ownerclan Similar Finder API</h2>
    <ul>
      <li>GET /openapi.json</li>
      <li>POST /search</li>
      <li>POST /export_csv</li>
      <li>GET /download/result.csv</li>
      <li>GET /privacy</li>
    </ul>
    """


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/search")
def search(req: SearchReq):
    # 1) seed 이미지/타이틀
    seed_title, seed_imgs = collect_seed_images(req.url, req.mode, req.only_first_image)

    # 2) 후보 URL 수집 (A: 상품명 기반 검색)
    keyword = seed_title
    # 너무 길면 검색이 깨질 수 있어 일부만
    keyword = _clean_text(keyword)[:60] if keyword else ""

    candidates = []
    if keyword:
        candidates = collect_candidate_urls_by_keyword(keyword, req.top_pages, req.max_candidates)

    # 3) seed_search_urls가 있으면 후보에 합치기
    if req.seed_search_urls:
        for u in req.seed_search_urls:
            if u and _is_ownerclan_product_url(u) and u not in candidates:
                candidates.append(u)
                if len(candidates) >= req.max_candidates:
                    break

    # 4) 비교
    results, debug = compare_and_rank(
        seed_imgs=seed_imgs,
        candidate_urls=candidates,
        phash_threshold=req.phash_threshold,
        only_first=req.only_first_image,
        mode=req.mode,
        seed_url=req.url,
    )

    payload = {
        "query_url": req.url,
        "seed_title": seed_title,
        "keyword_used": keyword,
        "mode": req.mode,
        "top_pages": req.top_pages,
        "phash_threshold": req.phash_threshold,
        "max_candidates": req.max_candidates,
        "seed_images_used": seed_imgs[: (1 if req.only_first_image else len(seed_imgs))],
        "candidate_urls_collected": len(candidates),
        "result_count": len(results),
        "results": results,
        "status": "ok",
        "debug": debug,
    }

    # CSV 준비(검색 때마다 최신으로 갱신)
    global LAST_CSV_BYTES, LAST_CSV_TS
    LAST_CSV_BYTES = make_csv_bytes(req.url, seed_title, results)
    LAST_CSV_TS = time.time()

    return JSONResponse(payload)


@app.post("/export_csv")
def export_csv(req: SearchReq):
    """
    export_csv는 내부적으로 /search를 한 번 실행해서 최신 결과 CSV를 만든 뒤,
    download_url을 돌려준다.
    """
    _ = search(req)  # 최신 결과 생성 (LAST_CSV_BYTES 갱신)
    return {
        "ok": True,
        "download_url": "/download/result.csv",
        "note": "Open /download/result.csv to download the latest CSV result."
    }


@app.get("/download/result.csv")
def download_csv():
    if not LAST_CSV_BYTES:
        return Response(content="No CSV generated yet. Call POST /search first.", media_type="text/plain", status_code=400)
    return Response(
        content=LAST_CSV_BYTES,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=result.csv"},
    )


@app.get("/privacy", response_class=HTMLResponse)
def privacy():
    return """
    <h1>Privacy Policy</h1>
    <p>This service processes user-provided URLs to fetch publicly available web pages and images for similarity comparison.</p>
    <p>No personal data is intentionally collected. Search results may be temporarily stored in memory to generate CSV downloads.</p>
    """


# -----------------------------
# Uvicorn entry (Railway)
# -----------------------------
# Railway는 PORT 환경변수를 주는 경우가 많음.
# Procfile 없이도 실행되게 해둠.
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
