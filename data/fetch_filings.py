from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datetime import datetime, timezone
from pathlib import Path

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from data import store
from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

UTC = timezone.utc

_FORMS = ["10-K", "10-Q"]


def _index_downloaded(ticker: str, form: str, dest: Path) -> list[dict]:
    """Walk downloaded filing dirs and build metadata records for the store index."""
    records = []
    # sec-edgar-downloader saves to: <download_folder>/<ticker>/<form>/ or similar structure
    # Try both naming conventions
    candidates = [
        dest / ticker / form,
        dest / ticker / form.replace("-", ""),
    ]
    form_dir = next((p for p in candidates if p.exists()), None)

    if form_dir is None or not form_dir.is_dir():
        logger.debug("No %s filings directory found for %s at %s", form, ticker, dest)
        return records

    for accession_dir in form_dir.iterdir():
        if not accession_dir.is_dir():
            continue
        accession_no = accession_dir.name

        # Infer filed_at from directory mtime (fallback; precise date requires metadata.json)
        filed_at = None
        metadata_file = accession_dir / "filing-details.json"
        if metadata_file.exists():
            try:
                import json
                meta = json.loads(metadata_file.read_text())
                filed_at_str = meta.get("filedAt") or meta.get("filed")
                if filed_at_str:
                    import dateutil.parser
                    filed_at = dateutil.parser.parse(filed_at_str).astimezone(UTC)
            except Exception:
                logger.debug("Could not parse filing-details.json for %s/%s", ticker, accession_no)

        if filed_at is None:
            # Use directory modification time as fallback
            filed_at = datetime.fromtimestamp(accession_dir.stat().st_mtime, tz=UTC)

        # Find the main filing document
        doc_files = list(accession_dir.glob("*.htm")) + list(accession_dir.glob("*.txt"))
        filing_path = str(doc_files[0]) if doc_files else str(accession_dir)

        records.append({
            "ticker": ticker,
            "form": form,
            "accession_no": accession_no,
            "filed_at": filed_at,
            "fetched_at": datetime.now(UTC),
            "path": filing_path,
        })

    return records


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=30),
    retry=retry_if_exception_type((ConnectionError, OSError)),
    reraise=True,
)
def _download_filings(ticker: str, form: str, dest: Path, limit: int) -> None:
    """Download filings for one ticker/form via sec-edgar-downloader."""
    from sec_edgar_downloader import Downloader
    settings = get_settings()

    # SEC requires a user-agent in "Name email" format
    parts = settings.sec_user_agent.split()
    company = " ".join(parts[:-1]) if len(parts) > 1 else "TradingAgent"
    email = parts[-1] if parts else "user@example.com"

    dl = Downloader(company, email, download_folder=str(dest))
    dl.get(form, ticker, limit=limit)


def fetch_filings(tickers: list[str] | None = None, *, limit: int = 2) -> dict[str, int]:
    """Download recent 10-K/10-Q per ticker, index metadata, return {ticker: filings_indexed}."""
    settings = get_settings()
    tickers = tickers or settings.tickers
    dest = settings.data_dir / "filings"
    dest.mkdir(parents=True, exist_ok=True)

    results: dict[str, int] = {}

    for ticker in tickers:
        ticker_count = 0
        all_records: list[dict] = []

        for form in _FORMS:
            try:
                _download_filings(ticker, form, dest, limit)
                records = _index_downloaded(ticker, form, dest)
                all_records.extend(records)
                logger.info("Indexed %d %s filings for %s", len(records), form, ticker)
            except Exception:
                logger.exception("Failed to download %s filings for %s", form, ticker)

        if all_records:
            store.save_filing_index(all_records)
            ticker_count = len(all_records)

        results[ticker] = ticker_count

    return results


if __name__ == "__main__":
    fetch_filings()
