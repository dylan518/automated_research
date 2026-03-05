import os
import re
import tempfile
import time
from typing import Any, Dict, Optional

import fitz
import requests


GRAPH_BASE_URL = "https://api.semanticscholar.org/graph/v1"
RECOMMENDATIONS_BASE_URL = "https://api.semanticscholar.org/recommendations/v1"


class SemanticScholarClient:
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 30) -> None:
        self.api_key = api_key or os.getenv("S2_KEY")
        if not self.api_key:
            raise ValueError("Missing S2_KEY. Set it in your environment or .env file.")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update({"x-api-key": self.api_key})

    def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        last_exc: Optional[Exception] = None
        for attempt in range(6):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout_seconds)
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as exc:
                last_exc = exc
                status = exc.response.status_code if exc.response is not None else None
                # Retry on rate limits with server hint if present.
                if status == 429 and attempt < 5:
                    retry_after_header = None
                    if exc.response is not None:
                        retry_after_header = exc.response.headers.get("Retry-After")
                    try:
                        retry_after_seconds = float(retry_after_header) if retry_after_header else None
                    except (TypeError, ValueError):
                        retry_after_seconds = None
                    delay = retry_after_seconds if retry_after_seconds is not None else (1.5 * (attempt + 1))
                    time.sleep(max(0.5, delay))
                    continue
                # Retry transient server errors only.
                if status is not None and 500 <= status < 600 and attempt < 5:
                    time.sleep(0.75 * (attempt + 1))
                    continue
                raise
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < 5:
                    time.sleep(0.75 * (attempt + 1))
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Unexpected request failure without exception.")

    def search_papers(self, query: str, limit: int = 10, year: Optional[str] = None) -> Dict[str, Any]:
        fields = ",".join(
            [
                "title",
                "authors",
                "year",
                "abstract",
                "url",
                "venue",
                "citationCount",
                "influentialCitationCount",
                "openAccessPdf",
                "externalIds",
            ]
        )
        params: Dict[str, Any] = {
            "query": query,
            "limit": max(1, min(limit, 50)),
            "fields": fields,
        }
        if year:
            params["year"] = year
        return self._get(f"{GRAPH_BASE_URL}/paper/search", params=params)

    def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        fields = ",".join(
            [
                "title",
                "authors",
                "year",
                "abstract",
                "url",
                "venue",
                "citationCount",
                "influentialCitationCount",
                "openAccessPdf",
                "externalIds",
                "references.title",
                "references.paperId",
                "citations.title",
                "citations.paperId",
            ]
        )
        return self._get(f"{GRAPH_BASE_URL}/paper/{paper_id}", params={"fields": fields})

    def recommend_papers(self, paper_id: str, limit: int = 10) -> Dict[str, Any]:
        fields = ",".join(
            [
                "title",
                "authors",
                "year",
                "abstract",
                "url",
                "venue",
                "citationCount",
                "openAccessPdf",
                "externalIds",
            ]
        )
        params = {"limit": max(1, min(limit, 50)), "fields": fields}
        return self._get(f"{RECOMMENDATIONS_BASE_URL}/papers/forpaper/{paper_id}", params=params)

    def get_open_access_pdf_url(self, paper_id: str) -> Optional[str]:
        details = self.get_paper_details(paper_id)
        pdf_obj = details.get("openAccessPdf") or {}
        return pdf_obj.get("url")

    @staticmethod
    def _fallback_pdf_url_from_details(details: Dict[str, Any]) -> Optional[str]:
        # Prefer explicit open access URL first.
        pdf_obj = details.get("openAccessPdf") or {}
        if isinstance(pdf_obj, dict):
            explicit = pdf_obj.get("url")
            if explicit:
                return explicit

        external_ids = details.get("externalIds") or {}
        if isinstance(external_ids, dict):
            arxiv_id = external_ids.get("ArXiv")
            if isinstance(arxiv_id, str) and arxiv_id.strip():
                return f"https://arxiv.org/pdf/{arxiv_id.strip()}.pdf"

        # Some records still have arXiv URLs in the paper URL.
        paper_url = details.get("url")
        if isinstance(paper_url, str) and "arxiv.org/abs/" in paper_url:
            return paper_url.replace("/abs/", "/pdf/") + ".pdf"
        return None

    @staticmethod
    def _is_url(value: str) -> bool:
        return value.startswith("http://") or value.startswith("https://")

    @staticmethod
    def _pdf_url_from_input(value: str) -> Optional[str]:
        raw = value.strip()
        if not raw:
            return None

        # Direct PDF links.
        if raw.endswith(".pdf"):
            return raw

        # arXiv abstract URLs.
        if "arxiv.org/abs/" in raw:
            base = raw.split("?", 1)[0].rstrip("/")
            return base.replace("/abs/", "/pdf/") + ".pdf"

        # arXiv ID passed directly (e.g., 2505.03335 or 2505.03335v3).
        if re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", raw):
            return f"https://arxiv.org/pdf/{raw}.pdf"

        return None

    def read_full_paper_text(self, paper_id: str, max_chars: int = 15000) -> Dict[str, Any]:
        input_pdf_url = self._pdf_url_from_input(paper_id)
        pdf_url: Optional[str] = input_pdf_url

        try:
            if not pdf_url:
                details = self.get_paper_details(paper_id)
                pdf_url = self._fallback_pdf_url_from_details(details)
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            return {
                "paper_id": paper_id,
                "success": False,
                "reason": "Failed to fetch paper details from Semantic Scholar.",
                "status_code": status,
            }
        except requests.RequestException as exc:
            return {
                "paper_id": paper_id,
                "success": False,
                "reason": f"Network error while fetching paper details: {exc}",
            }

        if not pdf_url:
            return {
                "paper_id": paper_id,
                "success": False,
                "reason": "No open-access or fallback PDF URL available for this paper.",
            }

        try:
            pdf_response = self.session.get(pdf_url, timeout=self.timeout_seconds)
            pdf_response.raise_for_status()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            return {
                "paper_id": paper_id,
                "success": False,
                "pdf_url": pdf_url,
                "reason": "Failed to download open-access PDF.",
                "status_code": status,
            }
        except requests.RequestException as exc:
            return {
                "paper_id": paper_id,
                "success": False,
                "pdf_url": pdf_url,
                "reason": f"Network error while downloading PDF: {exc}",
            }

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp_pdf:
                tmp_pdf.write(pdf_response.content)
                tmp_pdf.flush()

                doc = fitz.open(tmp_pdf.name)
                all_text = []
                for page in doc:
                    all_text.append(page.get_text("text"))
                doc.close()
        except Exception as exc:  # noqa: BLE001
            return {
                "paper_id": paper_id,
                "success": False,
                "pdf_url": pdf_url,
                "reason": f"Failed to parse PDF: {exc}",
            }

        full_text = "\n".join(all_text).strip()
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "\n\n...[truncated]..."

        return {
            "paper_id": paper_id,
            "success": True,
            "pdf_url": pdf_url,
            "text": full_text,
        }
