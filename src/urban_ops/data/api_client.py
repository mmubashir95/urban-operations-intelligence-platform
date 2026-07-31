"""Shared HTTPS JSON client for NYC Open Data API access."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from http.client import HTTPSConnection, RemoteDisconnected
from json import JSONDecodeError
from time import sleep
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
import json
import logging
import socket

from urban_ops.data.query_builder import QueryPlan


DNS_OVER_HTTPS_SERVER_NAME = "cloudflare-dns.com"
DNS_OVER_HTTPS_ADDRESSES = ["1.1.1.1", "1.0.0.1"]
DNS_OVER_HTTPS_TIMEOUT_SECONDS = 30
DEFAULT_USER_AGENT = "urban-operations-intelligence/1.0"
RETRYABLE_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}

_RESOLVED_HOST_ADDRESSES: dict[str, list[str]] = {}
LOGGER = logging.getLogger(__name__)


class APIRequestError(RuntimeError):
    """Raised when a JSON API request cannot be completed safely."""


class APIRetryExhaustedError(APIRequestError):
    """Raised after every permitted retryable request attempt fails."""


class FixedAddressHTTPError(APIRequestError):
    """HTTP failure returned through the TLS-verified DNS fallback path."""

    def __init__(self, message: str, *, status_code: int, retry_after: str | None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class PaginationIntegrityError(RuntimeError):
    """Raised when deterministic pagination invariants are violated."""


@dataclass(frozen=True)
class APIResponse:
    """Parsed response payload plus transport retry metadata."""

    payload: object
    retry_count: int


@dataclass(frozen=True)
class PageSummary:
    """Audit details for one successfully retrieved page."""

    page_number: int
    offset: int
    requested_limit: int
    returned_rows: int
    request_started_at: str
    request_completed_at: str
    retry_count: int


@dataclass(frozen=True)
class PaginationResult:
    """Complete records and integrity metadata from paginated retrieval."""

    records: list[dict[str, object]]
    pages: tuple[PageSummary, ...]
    retry_count: int
    duplicate_keys_crossing_pages: int


class FixedAddressHTTPSConnection(HTTPSConnection):
    """Open a TLS-verified HTTPS connection without a local DNS lookup."""

    def __init__(self, host: str, fixed_address: str, timeout: float) -> None:
        super().__init__(host, timeout=timeout)
        self.fixed_address = fixed_address

    def connect(self) -> None:
        """Connect to the fixed address while validating the host certificate."""
        self.sock = socket.create_connection(
            (self.fixed_address, self.port),
            self.timeout,
            self.source_address,
        )
        if self._tunnel_host:
            self._tunnel()
        self.sock = self._context.wrap_socket(
            self.sock,
            server_hostname=self.host,
        )


def _find_dns_error(error: BaseException) -> socket.gaierror | None:
    """Return a nested DNS-resolution error when one caused the failure."""
    current_error: BaseException | None = error
    visited_error_ids: set[int] = set()

    while current_error is not None and id(current_error) not in visited_error_ids:
        visited_error_ids.add(id(current_error))
        if isinstance(current_error, socket.gaierror):
            return current_error
        if isinstance(current_error, URLError) and isinstance(
            current_error.reason,
            BaseException,
        ):
            current_error = current_error.reason
            continue
        current_error = current_error.__cause__ or current_error.__context__

    return None


def _request_headers(headers: Mapping[str, str] | None) -> dict[str, str]:
    """Return request headers with stable JSON defaults."""
    request_headers = {
        "Accept": "application/json",
        "User-Agent": DEFAULT_USER_AGENT,
    }
    if headers:
        request_headers.update(headers)
    return request_headers


def _sanitize_error_text(text: str, headers: Mapping[str, str] | None) -> str:
    """Remove configured secret header values from an error message."""
    sanitized_text = text
    for header_name, header_value in (headers or {}).items():
        if not header_value:
            continue
        if header_name.lower() in {"x-app-token", "authorization"}:
            sanitized_text = sanitized_text.replace(str(header_value), "[REDACTED]")
    return sanitized_text


def _decode_json_response(body: bytes, *, context: str) -> object:
    """Decode one JSON response body with a useful error message."""
    try:
        return json.loads(body)
    except JSONDecodeError as error:
        raise APIRequestError(f"Malformed JSON response from {context}.") from error


def _response_body(response: object) -> bytes:
    """Read bytes from urlopen or HTTPSConnection response objects."""
    body = response.read()
    if isinstance(body, str):
        return body.encode("utf-8")
    return body


def _backoff_seconds(attempt: int) -> int:
    """Return bounded exponential retry delay for a one-indexed attempt."""
    return 2 ** (attempt - 1)


def _retry_delay(
    error: HTTPError | None,
    *,
    attempt: int,
    initial_backoff_seconds: float,
    maximum_backoff_seconds: float,
) -> float:
    """Return Retry-After seconds when valid, otherwise bounded backoff."""
    if error is not None and error.headers is not None:
        retry_after = error.headers.get("Retry-After")
        if retry_after:
            try:
                return min(max(float(retry_after), 0.0), maximum_backoff_seconds)
            except ValueError:
                pass
    return min(
        initial_backoff_seconds * (2 ** (attempt - 1)),
        maximum_backoff_seconds,
    )


def resolve_ipv4_with_dns_over_https(host: str) -> list[str]:
    """Resolve public IPv4 addresses through TLS-verified DNS-over-HTTPS."""
    if host in _RESOLVED_HOST_ADDRESSES:
        return _RESOLVED_HOST_ADDRESSES[host]

    request_path = "/dns-query?" + urlencode({"name": host, "type": "A"})
    last_error: BaseException | None = None

    for resolver_address in DNS_OVER_HTTPS_ADDRESSES:
        connection = FixedAddressHTTPSConnection(
            DNS_OVER_HTTPS_SERVER_NAME,
            resolver_address,
            timeout=DNS_OVER_HTTPS_TIMEOUT_SECONDS,
        )
        try:
            connection.request(
                "GET",
                request_path,
                headers={
                    "Accept": "application/dns-json",
                    "User-Agent": DEFAULT_USER_AGENT,
                },
            )
            response = connection.getresponse()
            body = _response_body(response)
            if response.status != 200:
                raise APIRequestError(
                    "DNS-over-HTTPS returned HTTP "
                    f"{response.status} {response.reason} for host {host}."
                )

            payload = _decode_json_response(body, context=DNS_OVER_HTTPS_SERVER_NAME)
            if not isinstance(payload, dict):
                raise APIRequestError(
                    "DNS-over-HTTPS returned JSON that was not an object."
                )
            addresses = [
                str(answer["data"])
                for answer in payload.get("Answer", [])
                if isinstance(answer, dict)
                and answer.get("type") == 1
                and isinstance(answer.get("data"), str)
            ]
            for address in addresses:
                socket.inet_aton(address)
            if addresses:
                _RESOLVED_HOST_ADDRESSES[host] = list(dict.fromkeys(addresses))
                return _RESOLVED_HOST_ADDRESSES[host]
            raise APIRequestError(
                f"DNS-over-HTTPS returned no IPv4 addresses for {host}."
            )
        except (
            OSError,
            TimeoutError,
            APIRequestError,
            JSONDecodeError,
        ) as error:
            last_error = error
        finally:
            connection.close()

    raise APIRequestError(
        f"Unable to resolve {host} through encrypted DNS fallback."
    ) from last_error


def fetch_json_from_resolved_addresses(
    url: str,
    host: str,
    addresses: list[str],
    timeout_seconds: float,
    *,
    headers: Mapping[str, str] | None = None,
) -> object:
    """Fetch JSON from fixed addresses while preserving host TLS checks."""
    parsed_url = urlsplit(url)
    request_target = parsed_url.path or "/"
    if parsed_url.query:
        request_target += f"?{parsed_url.query}"

    request_headers = _request_headers(headers)
    last_error: BaseException | None = None

    for address in addresses:
        connection = FixedAddressHTTPSConnection(
            host,
            address,
            timeout=timeout_seconds,
        )
        try:
            connection.request("GET", request_target, headers=request_headers)
            response = connection.getresponse()
            body = _response_body(response)
            if response.status >= 400:
                error_text = body.decode("utf-8", errors="replace")[:1000]
                error_text = _sanitize_error_text(error_text, headers)
                get_header = getattr(response, "getheader", lambda name: None)
                raise FixedAddressHTTPError(
                    f"HTTP {response.status} {response.reason} from {host}: "
                    f"{error_text}",
                    status_code=response.status,
                    retry_after=get_header("Retry-After"),
                )
            return _decode_json_response(body, context=host)
        except (
            OSError,
            TimeoutError,
            APIRequestError,
            JSONDecodeError,
        ) as error:
            last_error = error
        finally:
            connection.close()

    if isinstance(last_error, FixedAddressHTTPError):
        raise last_error
    raise APIRequestError(f"Unable to reach {host} by resolved address.") from last_error


def fetch_json_response(
    url: str,
    *,
    timeout_seconds: float,
    max_attempts: int,
    headers: Mapping[str, str] | None = None,
    initial_backoff_seconds: float = 1,
    maximum_backoff_seconds: float = 30,
) -> APIResponse:
    """Return parsed JSON and retry count with bounded retry behavior."""
    host = urlsplit(url).hostname
    if not host:
        raise ValueError(f"URL does not contain a host: {url}")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    if initial_backoff_seconds < 0 or maximum_backoff_seconds < 0:
        raise ValueError("Backoff settings must be non-negative.")

    request_headers = _request_headers(headers)
    last_error: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        retryable_http_error: HTTPError | None = None
        request = Request(url, headers=request_headers)
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return APIResponse(
                    _decode_json_response(
                        _response_body(response), context=f"{host} attempt {attempt}"
                    ),
                    retry_count=attempt - 1,
                )
        except HTTPError as error:
            error_body = error.read().decode("utf-8", errors="replace")[:2000]
            error_body = _sanitize_error_text(error_body, headers)
            http_error = APIRequestError(
                f"HTTP {error.code} {error.reason} from {host}: {error_body}"
            )
            if error.code not in RETRYABLE_HTTP_STATUS_CODES:
                raise http_error from error
            last_error = http_error
            retryable_http_error = error
        except (
            URLError,
            TimeoutError,
            socket.timeout,
            RemoteDisconnected,
            ConnectionResetError,
        ) as error:
            last_error = error
            if _find_dns_error(error) is not None:
                try:
                    addresses = resolve_ipv4_with_dns_over_https(host)
                    return APIResponse(
                        fetch_json_from_resolved_addresses(
                            url, host, addresses, timeout_seconds, headers=headers,
                        ),
                        retry_count=attempt - 1,
                    )
                except FixedAddressHTTPError as fallback_error:
                    if fallback_error.status_code not in RETRYABLE_HTTP_STATUS_CODES:
                        raise fallback_error
                    last_error = fallback_error
                    if fallback_error.retry_after:
                        retryable_http_error = HTTPError(
                            url,
                            fallback_error.status_code,
                            "DNS fallback HTTP error",
                            {"Retry-After": fallback_error.retry_after},
                            None,
                        )
                except APIRequestError as fallback_error:
                    last_error = fallback_error
        if attempt < max_attempts:
            delay = _retry_delay(
                retryable_http_error,
                attempt=attempt,
                initial_backoff_seconds=initial_backoff_seconds,
                maximum_backoff_seconds=maximum_backoff_seconds,
            )
            LOGGER.warning(
                "Retrying JSON request host=%s attempt=%s delay_seconds=%s",
                host, attempt + 1, delay,
            )
            sleep(delay)

    raise APIRetryExhaustedError(
        f"Request to {host} failed after {max_attempts} attempts "
        f"with timeout {timeout_seconds}: {last_error}"
    ) from last_error


def fetch_json_url(
    url: str,
    *,
    timeout_seconds: float,
    max_attempts: int,
    headers: Mapping[str, str] | None = None,
) -> object:
    """Return parsed JSON while preserving the original public client API."""
    return fetch_json_response(
        url,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        headers=headers,
    ).payload


def fetch_paginated_records(
    plan: QueryPlan,
    *,
    base_url: str,
    timeout_seconds: float,
    max_attempts: int,
    headers: Mapping[str, str] | None = None,
    initial_backoff_seconds: float = 1,
    maximum_backoff_seconds: float = 30,
    max_pages: int = 10_000,
    request_json: Callable[..., APIResponse] = fetch_json_response,
) -> PaginationResult:
    """Retrieve all deterministic pages while enforcing pagination integrity."""
    records: list[dict[str, object]] = []
    summaries: list[PageSummary] = []
    seen_page_hashes: set[str] = set()
    seen_keys: set[str] = set()
    crossing_duplicate_keys: set[str] = set()
    previous_order_key: tuple[str, str] | None = None
    offset = 0

    for page_number in range(1, max_pages + 1):
        started = datetime.now(timezone.utc)
        response = request_json(
            plan.page_url(base_url, offset=offset),
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            headers=headers,
            initial_backoff_seconds=initial_backoff_seconds,
            maximum_backoff_seconds=maximum_backoff_seconds,
        )
        completed = datetime.now(timezone.utc)
        payload = response.payload
        if not isinstance(payload, list) or not all(
            isinstance(record, dict) for record in payload
        ):
            raise APIRequestError("NYC Open Data page response must be a list of objects.")
        page = list(payload)
        if len(page) > plan.page_size:
            raise PaginationIntegrityError(
                f"Page {page_number} returned {len(page)} rows for limit {plan.page_size}."
            )
        page_hash = sha256(
            json.dumps(page, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if page and page_hash in seen_page_hashes:
            raise PaginationIntegrityError(f"Repeated page detected at offset {offset}.")
        if page:
            seen_page_hashes.add(page_hash)
        for record in page:
            order_key = (
                str(record.get("created_date", "")),
                str(record.get("unique_key", "")),
            )
            if previous_order_key is not None and order_key < previous_order_key:
                raise PaginationIntegrityError(
                    f"Unstable page ordering detected at offset {offset}: {order_key}."
                )
            previous_order_key = order_key
            unique_key = record.get("unique_key")
            if unique_key is not None:
                key = str(unique_key)
                if key in seen_keys:
                    crossing_duplicate_keys.add(key)
                seen_keys.add(key)
        summaries.append(
            PageSummary(
                page_number=page_number,
                offset=offset,
                requested_limit=plan.page_size,
                returned_rows=len(page),
                request_started_at=started.isoformat(),
                request_completed_at=completed.isoformat(),
                retry_count=response.retry_count,
            )
        )
        LOGGER.info(
            "Retrieved ingestion page page_number=%s offset=%s rows=%s retries=%s",
            page_number, offset, len(page), response.retry_count,
        )
        records.extend(page)
        if len(page) < plan.page_size:
            return PaginationResult(
                records=records,
                pages=tuple(summaries),
                retry_count=sum(item.retry_count for item in summaries),
                duplicate_keys_crossing_pages=len(crossing_duplicate_keys),
            )
        expected_next_offset = offset + plan.page_size
        if expected_next_offset <= offset:
            raise PaginationIntegrityError("Pagination offset did not progress.")
        offset = expected_next_offset
    raise PaginationIntegrityError(f"Pagination exceeded the {max_pages}-page safety limit.")
