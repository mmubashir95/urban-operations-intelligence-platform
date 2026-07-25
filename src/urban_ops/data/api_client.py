"""Shared HTTPS JSON client for NYC Open Data API access."""

from __future__ import annotations

from http.client import HTTPSConnection, RemoteDisconnected
from json import JSONDecodeError
from time import sleep
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
import json
import socket


DNS_OVER_HTTPS_SERVER_NAME = "cloudflare-dns.com"
DNS_OVER_HTTPS_ADDRESSES = ["1.1.1.1", "1.0.0.1"]
DNS_OVER_HTTPS_TIMEOUT_SECONDS = 30
DEFAULT_USER_AGENT = "urban-operations-intelligence/1.0"
RETRYABLE_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}

_RESOLVED_HOST_ADDRESSES: dict[str, list[str]] = {}


class APIRequestError(RuntimeError):
    """Raised when a JSON API request cannot be completed safely."""


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
                raise APIRequestError(
                    f"HTTP {response.status} {response.reason} from {host}: "
                    f"{error_text}"
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

    raise APIRequestError(
        f"Unable to reach {host} by resolved address."
    ) from last_error


def fetch_json_url(
    url: str,
    *,
    timeout_seconds: float,
    max_attempts: int,
    headers: Mapping[str, str] | None = None,
) -> object:
    """Return JSON using bounded retries and DNS-over-HTTPS as fallback only."""
    host = urlsplit(url).hostname
    if not host:
        raise ValueError(f"URL does not contain a host: {url}")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    if host in _RESOLVED_HOST_ADDRESSES:
        return fetch_json_from_resolved_addresses(
            url,
            host,
            _RESOLVED_HOST_ADDRESSES[host],
            timeout_seconds,
            headers=headers,
        )

    request_headers = _request_headers(headers)
    last_error: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        request = Request(url, headers=request_headers)
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return _decode_json_response(
                    _response_body(response),
                    context=f"{host} attempt {attempt}",
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
        except (
            URLError,
            TimeoutError,
            socket.timeout,
            RemoteDisconnected,
            APIRequestError,
            JSONDecodeError,
        ) as error:
            last_error = error
            if _find_dns_error(error) is not None:
                try:
                    addresses = resolve_ipv4_with_dns_over_https(host)
                    return fetch_json_from_resolved_addresses(
                        url,
                        host,
                        addresses,
                        timeout_seconds,
                        headers=headers,
                    )
                except APIRequestError as fallback_error:
                    raise APIRequestError(
                        f"Local DNS could not resolve {host}, and encrypted "
                        "DNS fallback could not complete the request."
                    ) from fallback_error
        if attempt < max_attempts:
            sleep(_backoff_seconds(attempt))

    raise APIRequestError(
        f"Request to {host} failed after {max_attempts} attempts "
        f"with timeout {timeout_seconds}."
    ) from last_error
