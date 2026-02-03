if req.mode == "main-detail":
    return {
        "query_url": req.url,
        "mode": "main-detail-disabled",
        "top_pages": req.top_pages,
        "seed_search_urls": req.seed_search_urls,
        "results": [],
        "status": "disabled_temporarily"
    }
