def collect_candidate_urls_by_keyword(keyword: str, top_pages: int, max_candidates: int) -> List[str]:
    """
    오너클랜 검색 대신 카테고리 전체 페이지를 순회하며 상품 URL 수집
    """
    candidates = []
    seen = set()

    for page in range(1, top_pages + 1):
        url = f"https://www.ownerclan.com/V2/product/list.php?page={page}"

        try:
            html = fetch_html(url)
        except Exception:
            continue

        soup = BeautifulSoup(html, "lxml")

        for a in soup.select("a[href*='/V2/product/view.php?selfcode=']"):
            link = _absolute_url("https://www.ownerclan.com", a.get("href"))

            if link not in seen:
                seen.add(link)
                candidates.append(link)

                if len(candidates) >= max_candidates:
                    return candidates

        time.sleep(0.2)

    return candidates
