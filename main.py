import re
import io
import csv
import uuid
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse, parse_qs

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import imagehash

app = FastAPI()

# ====== 설정 ======
OWNERCLAN_DOMAIN = "www.ownerclan.com"
DEFAULT_SEED_SEARCH_URLS = [
    # 필요하면 여기에 네가 자주 쓰는 검색 URL을 추가해도 됨
    "https://www.ownerclan.com/V2/product/search.php?topSearchKeywordInfo=&topSearchKeyword=%EA%B2%A8%EC%9A%B8&topSearchType=all",
]

# 무관 이미지로 자주 등장하는 경로/키워드
BLOCK_PATTERNS = [
    "/icon", "/common", "/btn", "/logo", "/banner", "sprite", "loading", "blank", "noimage"
]

# pHash 거리 임계값 (낮을수록 엄격)
PHASH_THRESHOLD = 8

# 다운로드한 CSV를 잠깐 보관(간단 버전)
CSV_STORE: Dict[str, str] = {}

# ====== 요청 모델 ======
class SearchReq(BaseModel):
    url: str
    top_pages: int = 10             # 후보 페이지(검색 결과) 몇 페이지 볼지
    mode: str = "main+detail"       # main_only | main+detail
    seed_search_urls: Optional[List[str]] = None  # 지정 안 하면 DEFAULT_SEED_SEARCH_URLS 사용
    phash_threshold: int = PHASH_THRESHOLD

# ====== 유틸 ======
def is_ownerclan_url(u: str) -> bool:
    try:
        return urlparse(u).netloc.endswith("ownerclan.com")
    except:
        return False

def normalize_url(base: str, src: str) -> str:
    return urljoin(base, src)

def looks_unrelated_image(url: str) -> bool:
    u = url.lower()
    return any(p in u for p in BLOCK_PATTERNS)

def extract_selfcode(product_url: str) -> str:
    try:
        qs = parse_qs(urlparse(product_url).query)
        return qs.get("selfcode", [""])[0]
    except:
        return ""

def parse_money(text: str) -> int:
    # "12,340원" / "12,340" → 12340
    m = re.search(r"([0-9][0-9,]*)", text.replace(" ", ""))
    if not m:
        return 0
    return int(m.group(1).replace(",", ""))

async def fetch_text(client: httpx.AsyncClient, url: str) -> str:
    r = await client.get(url, timeout=20)
    r.raise_for_status()
    return r.text

async def fetch_image_bytes(client: httpx.AsyncClient, url: str) -> bytes:
    r = await client.get(url, timeout=20)
    r.raise_for_status()
    return r.content

def safe_open_image(img_bytes: bytes) -> Optional[Image.Image]:
    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return im
    except:
        return None

def get_phash(im: Image.Image) -> imagehash.ImageHash:
    return imagehash.phash(im)

def hamming(a: imagehash.ImageHash, b: imagehash.ImageHash) -> int:
    return (a - b)

def guess_price_shipping_from_html(html: str) -> Dict[str, int]:
    """
    오너클랜 페이지 구조는 상품마다 조금씩 다를 수 있어서
    '일반가' / '배송비' 키워드를 기준으로 최대한 보수적으로 추출.
    """
    # 일반가
    normal_price = 0
    m = re.search(r"일반가[^0-9]*([0-9][0-9,]*)", html)
    if m:
        normal_price = int(m.group(1).replace(",", ""))

    # 배송비
    shipping = 0
    m2 = re.search(r"배송비[^0-9]*([0-9][0-9,]*)", html)
    if m2:
        shipping = int(m2.group(1).replace(",", ""))

    return {"normal_price": normal_price, "shipping": shipping}

def pick_candidate_img_tags(soup: BeautifulSoup) -> List[str]:
    """
    가능한 한 '상품 상세'에 가까운 img src를 모으고,
    없으면 전체에서 모은 후 필터링.
    """
    srcs: List[str] = []

    # 1) 상세 설명로 보이는 영역 우선(흔한 패턴들 후보)
    for selector in [
        "div#content img",
        "div#productDetail img",
        "div.detail img",
        "div.view_content img",
        "div#goods_detail img"
    ]:
        found = soup.select(selector)
        for img in found:
            s = img.get("src") or img.get("data-src") or ""
            if s:
                srcs.append(s)

    # 2) 없으면 전체 img
    if not srcs:
        for img in soup.find_all("img"):
            s = img.get("src") or img.get("data-src") or ""
            if s:
                srcs.append(s)

    # 중복 제거(순서 유지)
    seen = set()
    uniq = []
    for s in srcs:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq

async def extract_product_images_phash(client: httpx.AsyncClient, product_url: str, mode: str) -> Dict[str, Any]:
    html = await fetch_text(client, product_url)
    soup = BeautifulSoup(html, "html.parser")

    raw_srcs = pick_candidate_img_tags(soup)
    img_urls: List[str] = []
    for s in raw_srcs:
        u = normalize_url(product_url, s)
        if not is_ownerclan_url(u):
            continue
        if looks_unrelated_image(u):
            continue
        img_urls.append(u)

    # mode에 따라 대표 1장만 쓰기
    if mode == "main_only" and img_urls:
        img_urls = img_urls[:1]

    # 이미지 다운로드 + 크기 필터 + phash 생성
    hashes: List[Dict[str, Any]] = []
    for u in img_urls:
        try:
            b = await fetch_image_bytes(client, u)
            im = safe_open_image(b)
            if im is None:
                continue
            w, h = im.size
            if w < 200 or h < 200:
                continue
            ph = get_phash(im)
            hashes.append({"url": u, "phash": ph, "w": w, "h": h})
        except:
            continue

    price_info = guess_price_shipping_from_html(html)
    return {
        "url": product_url,
        "selfcode": extract_selfcode(product_url),
        "images": hashes,
        "normal_price": price_info["normal_price"],
        "shipping": price_info["shipping"],
        "total_price": price_info["normal_price"] + price_info["shipping"],
    }

async def collect_candidate_product_urls(client: httpx.AsyncClient, seed_search_urls: List[str], top_pages: int) -> List[str]:
    """
    검색 결과 페이지(Seed URL)에서 상품 상세 링크들을 모음.
    top_pages만큼 페이지네이션을 시도(간단형).
    """
    product_urls = []
    seen = set()

    for seed in seed_search_urls:
        # 1) seed 페이지 읽기
        try:
            html = await fetch_text(client, seed)
        except:
            continue

        # 2) 상세 링크 추출( selfcode= 가 들어간 view.php 링크 )
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "product/view.php" in href and "selfcode=" in href:
                full = normalize_url(seed, href)
                if not is_ownerclan_url(full):
                    continue
                if full not in seen:
                    seen.add(full)
                    product_urls.append(full)

        # 3) 페이지네이션: seed URL에 page 파라미터를 붙여 추가 탐색(사이트마다 다를 수 있어 보수적으로)
        #    - 실패해도 괜찮게 설계
        base = seed
        for p in range(2, top_pages + 1):
            # page=2,3... 붙여보기
            sep = "&" if "?" in base else "?"
            page_url = f"{base}{sep}page={p}"
            try:
                html_p = await fetch_text(client, page_url)
            except:
                continue
            soup_p = BeautifulSoup(html_p, "html.parser")
            for a in soup_p.find_all("a", href=True):
                href = a["href"]
                if "product/view.php" in href and "selfcode=" in href:
                    full = normalize_url(page_url, href)
                    if not is_ownerclan_url(full):
                        continue
                    if full not in seen:
                        seen.add(full)
                        product_urls.append(full)

    return product_urls

def compare_products_by_phash(query_imgs: List[Dict[str, Any]], cand_imgs: List[Dict[str, Any]], threshold: int) -> Dict[str, Any]:
    """
    쿼리 이미지들 vs 후보 이미지들 전체 비교(정석형 B)
    - 어떤 한 쌍이라도 threshold 이하이면 "유사"로 간주
    - 최소 거리, 매칭 수 등을 반환
    """
    if not query_imgs or not cand_imgs:
        return {"is_similar": False, "min_dist": 999, "match_count": 0}

    min_dist = 999
    match_count = 0
    for qi in query_imgs:
        for ci in cand_imgs:
            d = hamming(qi["phash"], ci["phash"])
            if d < min_dist:
                min_dist = d
            if d <= threshold:
                match_count += 1

    return {"is_similar": match_count > 0, "min_dist": min_dist, "match_count": match_count}

def make_csv(rows: List[Dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "rank", "total_price", "normal_price", "shipping",
        "match_count", "min_dist", "selfcode", "product_url"
    ])
    for i, r in enumerate(rows, start=1):
        writer.writerow([
            i, r.get("total_price", 0), r.get("normal_price", 0), r.get("shipping", 0),
            r.get("match_count", 0), r.get("min_dist", 999),
            r.get("selfcode", ""), r.get("product_url", "")
        ])
    return output.getvalue()

# ====== API ======
@app.post("/search")
async def search(req: SearchReq):
    if not req.url:
        return {"error": "url is required"}

    if not is_ownerclan_url(req.url):
        return {"error": "Only ownerclan.com URLs are allowed."}

    seed_urls = req.seed_search_urls or DEFAULT_SEED_SEARCH_URLS

    async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) as client:
        # 1) 쿼리(입력) 상품 이미지 해시 만들기
        query = await extract_product_images_phash(client, req.url, req.mode)

        # 2) 후보 상품 URL 모으기
        cand_urls = await collect_candidate_product_urls(client, seed_urls, req.top_pages)

        # 3) 후보 상품들 분석 + 비교
        results = []
        for cu in cand_urls:
            # 입력 URL 자기 자신은 제외
            if cu == req.url:
                continue

            cand = await extract_product_images_phash(client, cu, req.mode)

            # ✅ 유사도 비교: 전체 이미지 vs 전체 이미지 (정석형)
            cmp = compare_products_by_phash(query["images"], cand["images"], req.phash_threshold)
            if not cmp["is_similar"]:
                continue

            results.append({
                "product_url": cand["url"],
                "selfcode": cand["selfcode"],
                "normal_price": cand["normal_price"],   # ✅ 일반가만
                "shipping": cand["shipping"],
                "total_price": cand["total_price"],     # ✅ 일반가+배송비
                "match_count": cmp["match_count"],
                "min_dist": cmp["min_dist"],
            })

        # 4) 가격 정렬(배송비 포함 최저가)
        results.sort(key=lambda x: (x.get("total_price", 0), x.get("min_dist", 999)))

        return {
            "query_url": req.url,
            "mode": req.mode,
            "top_pages": req.top_pages,
            "phash_threshold": req.phash_threshold,
            "result_count": len(results),
            "results": results,
            "status": "ok"
        }

@app.post("/export_csv")
async def export_csv(req: SearchReq):
    # search 결과를 그대로 CSV로 만들어서 다운로드 링크 제공(간단 저장)
    data = await search(req)
    if "results" not in data:
        return data

    csv_text = make_csv(data["results"])
    file_id = str(uuid.uuid4())
    CSV_STORE[file_id] = csv_text

    return {
        "ok": True,
        "file_id": file_id,
        "download_url": f"/download/{file_id}",
        "preview": csv_text[:5000]
    }

@app.get("/download/{file_id}")
def download(file_id: str):
    # 브라우저에서 열면 CSV 내용이 그대로 보임(저장 가능)
    csv_text = CSV_STORE.get(file_id)
    if not csv_text:
        return {"error": "file not found"}
    return csv_text

@app.get("/privacy")
def privacy():
    return {
        "policy": "This service receives only the URL you provide to search similar products. No personal data is stored."
    }
