"""Download and freeze a Core-CCR external replication dataset from JAO.

The downloader requests active constraints one delivery day at a time so the
top-level ``lastModifiedOn`` value can be attached to exactly the rows it
governs.  Raw responses are resumably cached, while the derived gzip CSVs are
written deterministically.  No API token is required for these public Core
publication endpoints.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd
import requests

try:
    from scripts.core_external_replication import (
        CORE_PHYSICAL_ZONES,
        CORE_SHADOW_ENDPOINT,
        CORE_SPREAD_ENDPOINT,
        atomic_json,
        canonicalize_shadow_rows,
        core_local_delivery_windows,
        infer_spread_convention,
        parse_utc,
        reconstruct_centered_price_field,
        sha256_file,
        write_deterministic_csv_gz,
    )
except ModuleNotFoundError:
    from core_external_replication import (
        CORE_PHYSICAL_ZONES,
        CORE_SHADOW_ENDPOINT,
        CORE_SPREAD_ENDPOINT,
        atomic_json,
        canonicalize_shadow_rows,
        core_local_delivery_windows,
        infer_spread_convention,
        parse_utc,
        reconstruct_centered_price_field,
        sha256_file,
        write_deterministic_csv_gz,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "external_core_replication"


def _request_page(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    *,
    retries: int,
    timeout_seconds: float,
) -> requests.Response:
    response: requests.Response | None = None
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=timeout_seconds)
        except requests.RequestException:
            if attempt + 1 == retries:
                raise
            time.sleep(min(2.0**attempt, 30.0))
            continue
        if response.status_code == 429:
            retry_after = max(float(response.headers.get("Retry-After", 5.0)), 1.0)
            time.sleep(retry_after)
            continue
        if 500 <= response.status_code < 600:
            time.sleep(min(2.0**attempt, 30.0))
            continue
        response.raise_for_status()
        return response
    status = response.status_code if response is not None else None
    raise RuntimeError(f"JAO Core request failed after retries: status={status}, url={url}")


def _fetch_day(
    session: requests.Session,
    endpoint: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_path: Path,
    *,
    page_size: int,
    retries: int,
    timeout_seconds: float,
    pause_seconds: float,
    force: bool,
) -> tuple[dict[str, Any], bool]:
    if cache_path.is_file() and not force:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(cached, dict) or not isinstance(cached.get("rows"), list):
            raise RuntimeError(f"Malformed Core cache: {cache_path}")
        return cached, True

    rows: list[dict[str, Any]] = []
    skip = 0
    last_modified: str | None = None
    page_count = 0
    while True:
        params = {
            "FromUtc": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ToUtc": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "Skip": skip,
            "Take": page_size,
        }
        response = _request_page(
            session,
            endpoint,
            params,
            retries=retries,
            timeout_seconds=timeout_seconds,
        )
        payload = response.json()
        page = payload.get("data", [])
        if not isinstance(page, list):
            raise RuntimeError(f"Unexpected JAO Core payload from {endpoint}")
        page_count += 1
        rows.extend(page)
        modified = payload.get("lastModifiedOn")
        if modified:
            if last_modified is not None and parse_utc(modified) != parse_utc(last_modified):
                raise RuntimeError(
                    f"JAO Core pagination changed lastModifiedOn within {start.date()}"
                )
            last_modified = str(modified)
        total = payload.get("totalRowsWithFilter", payload.get("totalRows"))
        if total is not None:
            if len(rows) >= int(total) or not page:
                break
        elif len(page) < page_size:
            break
        if not page:
            break
        skip += len(page)
        time.sleep(pause_seconds)

    cached = {
        "endpoint": endpoint,
        "from_utc": start.isoformat(),
        "to_utc": end.isoformat(),
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "lastModifiedOn": last_modified,
        "page_count": page_count,
        "rows": rows,
    }
    atomic_json(cache_path, cached, indent=None)
    time.sleep(pause_seconds)
    return cached, False


def _date_tag(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y%m%d")


def _payload_is_causal(payload: dict[str, Any]) -> bool:
    rows = payload.get("rows", [])
    if not rows:
        return True
    publication = payload.get("lastModifiedOn")
    if not publication:
        return False
    first_delivery = min(parse_utc(row["dateTimeUtc"]) for row in rows)
    return parse_utc(publication) <= first_delivery


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-10-01")
    parser.add_argument("--end", default="2025-10-01")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--page_size", type=int, default=5_000)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--timeout_seconds", type=float, default=90.0)
    parser.add_argument("--api_pause_seconds", type=float, default=0.75)
    parser.add_argument("--sign_audit_days", type=int, default=30)
    parser.add_argument(
        "--spread_convention",
        choices=("auto", "source_minus_destination", "destination_minus_source"),
        default="auto",
    )
    parser.add_argument("--force_download", action="store_true")
    args = parser.parse_args()
    if args.page_size < 100 or args.retries < 1 or args.timeout_seconds < 10.0:
        raise ValueError("Invalid page-size, retry, or timeout setting")
    if args.api_pause_seconds < 0.0 or args.sign_audit_days < 2:
        raise ValueError("Pause must be non-negative and sign audit at least two days")

    start = parse_utc(args.start)
    end = parse_utc(args.end)
    if start >= end or start.floor("D") != start or end.floor("D") != end:
        raise ValueError("--start/--end must be ordered UTC day boundaries")
    output_root = args.output_root.resolve()
    data_root = output_root / "data"
    cache_root = output_root / ".raw_cache"
    shadow_cache = cache_root / "shadowPrices"
    spread_cache = cache_root / "priceSpread"
    shadow_cache.mkdir(parents=True, exist_ok=True)
    spread_cache.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "AF-CDMO-Core-external-replication/1.0"})
    shadow_rows: list[dict[str, Any]] = []
    spread_rows: list[dict[str, Any]] = []
    causally_excluded_timestamps: set[pd.Timestamp] = set()
    shadow_requests: list[dict[str, Any]] = []
    shadow_windows = core_local_delivery_windows(start, end)
    for window_index, (cursor, next_cursor) in enumerate(shadow_windows, start=1):
        tag = f"{cursor.strftime('%Y%m%dT%H%M')}_{next_cursor.strftime('%Y%m%dT%H%M')}"
        shadow_payload, shadow_cached = _fetch_day(
            session,
            CORE_SHADOW_ENDPOINT,
            cursor,
            next_cursor,
            shadow_cache / f"shadow_localday_{tag}.json",
            page_size=args.page_size,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
            pause_seconds=args.api_pause_seconds,
            force=args.force_download,
        )
        publication = shadow_payload.get("lastModifiedOn")
        if shadow_payload["rows"] and not publication:
            raise RuntimeError(
                f"Core active constraints for {tag} omit top-level lastModifiedOn"
            )
        refined = False
        excluded_in_window: list[str] = []
        if _payload_is_causal(shadow_payload):
            for row in shadow_payload["rows"]:
                shadow_rows.append({**row, "_publication_utc": publication})
        else:
            refined = True
            delivery_intervals = sorted(
                {parse_utc(row["dateTimeUtc"]) for row in shadow_payload["rows"]}
            )
            for delivery_index, delivery in enumerate(delivery_intervals):
                interval_end = (
                    delivery_intervals[delivery_index + 1]
                    if delivery_index + 1 < len(delivery_intervals)
                    else next_cursor
                )
                if interval_end <= delivery:
                    raise RuntimeError("Core MTU refinement failed to advance")
                interval_tag = (
                    f"{delivery.strftime('%Y%m%dT%H%M')}_"
                    f"{interval_end.strftime('%Y%m%dT%H%M')}"
                )
                interval_cache = shadow_cache / f"shadow_mtu_{interval_tag}.json"
                legacy_hour_cache = shadow_cache / f"shadow_hour_{interval_tag}.json"
                if (
                    not interval_cache.exists()
                    and interval_end - delivery == pd.Timedelta(hours=1)
                    and legacy_hour_cache.exists()
                ):
                    interval_cache = legacy_hour_cache
                interval_payload, _ = _fetch_day(
                    session,
                    CORE_SHADOW_ENDPOINT,
                    delivery,
                    interval_end,
                    interval_cache,
                    page_size=args.page_size,
                    retries=args.retries,
                    timeout_seconds=args.timeout_seconds,
                    pause_seconds=args.api_pause_seconds,
                    force=args.force_download,
                )
                interval_publication = interval_payload.get("lastModifiedOn")
                if _payload_is_causal(interval_payload):
                    for row in interval_payload["rows"]:
                        shadow_rows.append(
                            {**row, "_publication_utc": interval_publication}
                        )
                else:
                    causally_excluded_timestamps.add(delivery)
                    excluded_in_window.append(delivery.isoformat())
        shadow_requests.append(
            {
                "from_utc": cursor.isoformat(),
                "to_utc": next_cursor.isoformat(),
                "delivery_timezone": "Europe/Brussels",
                "rows": len(shadow_payload["rows"]),
                "last_modified_utc": publication,
                "cached": shadow_cached,
                "mtu_refinement_required": refined,
                "causally_excluded_delivery_intervals": excluded_in_window,
            }
        )
        if window_index == 1 or window_index % 14 == 0 or window_index == len(shadow_windows):
            print(
                f"[CORE-SHADOW] windows={window_index}/{len(shadow_windows)} "
                f"through={next_cursor.isoformat()} rows={len(shadow_rows):,}",
                flush=True,
            )

    spread_requests: list[dict[str, Any]] = []
    cursor = start
    day_index = 0
    while cursor < end:
        next_cursor = min(cursor + pd.Timedelta(days=1), end)
        tag = _date_tag(cursor)
        spread_payload, spread_cached = _fetch_day(
            session,
            CORE_SPREAD_ENDPOINT,
            cursor,
            next_cursor,
            spread_cache / f"spread_{tag}.json",
            page_size=args.page_size,
            retries=args.retries,
            timeout_seconds=args.timeout_seconds,
            pause_seconds=args.api_pause_seconds,
            force=args.force_download,
        )
        spread_rows.extend(spread_payload["rows"])
        spread_requests.append(
            {
                "from_utc": cursor.isoformat(),
                "to_utc": next_cursor.isoformat(),
                "rows": len(spread_payload["rows"]),
                "cached": spread_cached,
            }
        )
        day_index += 1
        if day_index == 1 or day_index % 14 == 0 or next_cursor == end:
            print(
                f"[CORE-SPREAD] days={day_index} through={next_cursor.date()} "
                f"rows={len(spread_rows):,}",
                flush=True,
            )
        cursor = next_cursor

    if not shadow_rows or not spread_rows:
        raise RuntimeError("Core endpoints returned no usable rows")
    dual, dual_audit = canonicalize_shadow_rows(shadow_rows)
    spread_raw = pd.DataFrame.from_records(spread_rows)
    audit_end = min(start + pd.Timedelta(days=args.sign_audit_days), end)
    if args.spread_convention == "auto":
        selected, sign_audit, price = infer_spread_convention(
            spread_raw,
            dual,
            audit_end=audit_end,
            minimum_overlap=min(24, max(2, int((audit_end - start) / pd.Timedelta(hours=1)) // 2)),
        )
    else:
        selected = args.spread_convention
        price, diagnostics = reconstruct_centered_price_field(
            spread_raw, convention=selected
        )
        sign_audit = {
            "selected": selected,
            "purpose": "explicit user-specified external API schema orientation",
            "field_diagnostics": {selected: diagnostics.__dict__},
        }

    if causally_excluded_timestamps:
        price["delivery_utc"] = pd.to_datetime(
            price["delivery_utc"], utc=True, errors="raise", format="mixed"
        )
        price = price.loc[
            ~price["delivery_utc"].isin(causally_excluded_timestamps)
        ].copy()
        if price.empty:
            raise RuntimeError("All Core target hours were excluded by causal timing")

    period = f"{_date_tag(start)}_{_date_tag(end)}"
    dual_path = data_root / f"CoreBindingDualMeasure_{period}.csv.gz"
    price_path = data_root / f"CoreCenteredPriceField_{period}.csv.gz"
    dual["delivery_utc"] = dual["delivery_utc"].map(lambda value: parse_utc(value).isoformat())
    dual["publication_utc"] = dual["publication_utc"].map(
        lambda value: parse_utc(value).isoformat()
    )
    price["delivery_utc"] = price["delivery_utc"].map(
        lambda value: parse_utc(value).isoformat()
    )
    write_deterministic_csv_gz(dual, dual_path)
    write_deterministic_csv_gz(price, price_path)

    request_path = output_root / f"core_request_audit_{period}.json"
    request_manifest = {
        "shadow_delivery_day_windows": shadow_requests,
        "spread_utc_day_windows": spread_requests,
    }
    atomic_json(request_path, request_manifest)
    manifest = {
        "protocol": "core_ccr_external_replication_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "period": {"start_utc": start.isoformat(), "end_utc_exclusive": end.isoformat()},
        "source": {
            "active_constraints": CORE_SHADOW_ENDPOINT,
            "price_spreads": CORE_SPREAD_ENDPOINT,
            "provider": "Joint Allocation Office (JAO)",
        },
        "scope": {
            "physical_zones": list(CORE_PHYSICAL_ZONES),
            "geometry": "global net-position balance only (rank 1)",
            "excluded_claim": (
                "No claim that public Core tables identify the full four technical "
                "equalities or ALEGrO virtual-hub equalities"
            ),
        },
        "dual_audit": dual_audit,
        "causal_revision_audit": {
            "policy": (
                "Refine non-causal local-day batches by published MTU; exclude "
                "intervals whose "
                "response-level lastModifiedOn remains later than delivery"
            ),
            "refined_local_day_windows": int(
                sum(item["mtu_refinement_required"] for item in shadow_requests)
            ),
            "excluded_delivery_interval_count": len(causally_excluded_timestamps),
            "excluded_delivery_intervals_utc": [
                value.isoformat() for value in sorted(causally_excluded_timestamps)
            ],
            "retained_price_hour_count": int(len(price)),
        },
        "spread_sign_audit": sign_audit,
        "selected_spread_convention": selected,
        "artifacts": {
            "dual": {
                "path": str(dual_path),
                "sha256": sha256_file(dual_path),
                "rows": int(len(dual)),
            },
            "centered_price_field": {
                "path": str(price_path),
                "sha256": sha256_file(price_path),
                "rows": int(len(price)),
            },
            "request_audit": {
                "path": str(request_path),
                "sha256": sha256_file(request_path),
                "shadow_windows": len(shadow_requests),
                "spread_windows": len(spread_requests),
            },
        },
        "raw_cache": {
            "path": str(cache_root),
            "redistribution": "local only; excluded from the public release",
        },
    }
    manifest_path = output_root / f"core_external_manifest_{period}.json"
    atomic_json(manifest_path, manifest)
    print(f"[OK] Core dual measure: {dual_path}", flush=True)
    print(f"[OK] Core centered price field: {price_path}", flush=True)
    print(f"[OK] Core manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
