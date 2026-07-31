from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
import io
import socket
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from urban_ops.data import api_client
from urban_ops.data.query_builder import build_query_plan


@pytest.fixture(autouse=True)
def clear_resolved_address_cache() -> None:
    """Keep the client's process cache isolated between unit tests."""
    api_client._RESOLVED_HOST_ADDRESSES.clear()
    yield
    api_client._RESOLVED_HOST_ADDRESSES.clear()


class FakeUrlopenResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "FakeUrlopenResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class FakeHTTPResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status: int = 200,
        reason: str = "OK",
    ) -> None:
        self._body = body
        self.status = status
        self.reason = reason

    def read(self) -> bytes:
        return self._body


def test_fetch_json_url_returns_decoded_json(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    def fake_urlopen(request: object, timeout: float) -> FakeUrlopenResponse:
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        return FakeUrlopenResponse(b'{"ok": true}')

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)

    payload = api_client.fetch_json_url(
        "https://example.test/resource.json",
        timeout_seconds=12,
        max_attempts=1,
    )

    assert payload == {"ok": True}
    assert captured["timeout"] == 12
    assert captured["headers"]["Accept"] == "application/json"
    assert captured["headers"]["User-agent"] == api_client.DEFAULT_USER_AGENT


def test_fetch_json_url_preserves_optional_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_urlopen(request: object, timeout: float) -> FakeUrlopenResponse:
        captured["headers"] = dict(request.header_items())
        return FakeUrlopenResponse(b"[]")

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)

    api_client.fetch_json_url(
        "https://example.test/resource.json",
        timeout_seconds=12,
        max_attempts=1,
        headers={
            "User-Agent": "custom-agent/1.0",
            "X-App-Token": "secret-token",
        },
    )

    assert captured["headers"]["User-agent"] == "custom-agent/1.0"
    assert captured["headers"]["X-app-token"] == "secret-token"
    assert captured["headers"]["Accept"] == "application/json"


def test_fetch_json_url_retries_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0, "sleeps": []}

    def fake_urlopen(request: object, timeout: float) -> FakeUrlopenResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            raise URLError("temporary failure")
        return FakeUrlopenResponse(b'{"ok": true}')

    def fake_sleep(seconds: int) -> None:
        calls["sleeps"].append(seconds)

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    monkeypatch.setattr(api_client, "sleep", fake_sleep)

    payload = api_client.fetch_json_url(
        "https://example.test/resource.json",
        timeout_seconds=12,
        max_attempts=2,
    )

    assert payload == {"ok": True}
    assert calls == {"count": 2, "sleeps": [1]}


def test_fetch_json_url_raises_after_exhausted_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def fake_urlopen(request: object, timeout: float) -> FakeUrlopenResponse:
        calls["count"] += 1
        raise URLError("temporary failure")

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    monkeypatch.setattr(api_client, "sleep", lambda seconds: None)

    with pytest.raises(api_client.APIRequestError, match="failed after 3 attempts"):
        api_client.fetch_json_url(
            "https://example.test/resource.json",
            timeout_seconds=12,
            max_attempts=3,
        )

    assert calls["count"] == 3


def test_fetch_json_url_rejects_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_client,
        "urlopen",
        lambda request, timeout: FakeUrlopenResponse(b"{not json"),
    )

    with pytest.raises(api_client.APIRequestError, match="Malformed JSON"):
        api_client.fetch_json_url(
            "https://example.test/resource.json",
            timeout_seconds=12,
            max_attempts=1,
        )


def test_fetch_json_url_raises_non_retryable_http_error_without_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(request: object, timeout: float) -> FakeUrlopenResponse:
        raise HTTPError(
            "https://example.test/resource.json",
            404,
            "Not Found",
            {},
            io.BytesIO(b"missing secret-token"),
        )

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)

    with pytest.raises(api_client.APIRequestError) as error:
        api_client.fetch_json_url(
            "https://example.test/resource.json",
            timeout_seconds=12,
            max_attempts=3,
            headers={"X-App-Token": "secret-token"},
        )

    assert "HTTP 404 Not Found" in str(error.value)
    assert "secret-token" not in str(error.value)


def test_fetch_json_url_retries_retryable_http_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0, "sleeps": []}

    def fake_urlopen(request: object, timeout: float) -> FakeUrlopenResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            raise HTTPError(
                "https://example.test/resource.json",
                503,
                "Service Unavailable",
                {},
                io.BytesIO(b"try again later"),
            )
        return FakeUrlopenResponse(b'{"ok": true}')

    def fake_sleep(seconds: int) -> None:
        calls["sleeps"].append(seconds)

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    monkeypatch.setattr(api_client, "sleep", fake_sleep)

    payload = api_client.fetch_json_url(
        "https://example.test/resource.json",
        timeout_seconds=12,
        max_attempts=2,
    )

    assert payload == {"ok": True}
    assert calls == {"count": 2, "sleeps": [1]}


def test_fetch_json_url_rejects_non_positive_max_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        api_client.fetch_json_url(
            "https://example.test/resource.json",
            timeout_seconds=12,
            max_attempts=0,
        )


def test_dns_fallback_runs_after_local_dns_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_calls = {}

    def fake_urlopen(request: object, timeout: float) -> FakeUrlopenResponse:
        raise URLError(socket.gaierror("nodename nor servname provided"))

    def fake_resolve(host: str) -> list[str]:
        fallback_calls["host"] = host
        return ["203.0.113.10"]

    def fake_resolved_fetch(
        url: str,
        host: str,
        addresses: list[str],
        timeout_seconds: float,
        *,
        headers: dict[str, str] | None = None,
    ) -> object:
        fallback_calls["fetch"] = (url, host, addresses, timeout_seconds, headers)
        return [{"fallback": True}]

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    monkeypatch.setattr(api_client, "resolve_ipv4_with_dns_over_https", fake_resolve)
    monkeypatch.setattr(
        api_client,
        "fetch_json_from_resolved_addresses",
        fake_resolved_fetch,
    )

    payload = api_client.fetch_json_url(
        "https://data.cityofnewyork.us/resource/erm2-nwe9.json",
        timeout_seconds=20,
        max_attempts=3,
        headers={"X-App-Token": "secret-token"},
    )

    assert payload == [{"fallback": True}]
    assert fallback_calls["host"] == "data.cityofnewyork.us"
    assert fallback_calls["fetch"] == (
        "https://data.cityofnewyork.us/resource/erm2-nwe9.json",
        "data.cityofnewyork.us",
        ["203.0.113.10"],
        20,
        {"X-App-Token": "secret-token"},
    )


def test_fixed_address_fetch_preserves_host_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class FakeConnection:
        def __init__(self, host: str, fixed_address: str, timeout: float) -> None:
            captured["init"] = (host, fixed_address, timeout)

        def request(self, method: str, target: str, headers: dict[str, str]) -> None:
            captured["request"] = (method, target, headers)

        def getresponse(self) -> FakeHTTPResponse:
            return FakeHTTPResponse(b'{"ok": true}')

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(api_client, "FixedAddressHTTPSConnection", FakeConnection)

    payload = api_client.fetch_json_from_resolved_addresses(
        "https://data.cityofnewyork.us/resource/erm2-nwe9.json?$limit=1",
        "data.cityofnewyork.us",
        ["203.0.113.10"],
        20,
    )

    assert payload == {"ok": True}
    assert captured["init"] == ("data.cityofnewyork.us", "203.0.113.10", 20)
    assert captured["request"][0] == "GET"
    assert captured["request"][1] == "/resource/erm2-nwe9.json?$limit=1"
    assert captured["request"][2]["User-Agent"] == api_client.DEFAULT_USER_AGENT
    assert captured["closed"] is True


def test_fixed_address_connection_uses_original_hostname_for_tls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class FakeContext:
        def wrap_socket(self, sock: object, server_hostname: str) -> object:
            captured["server_hostname"] = server_hostname
            return ("tls", sock, server_hostname)

    monkeypatch.setattr(
        api_client.socket,
        "create_connection",
        lambda address, timeout, source_address: captured.setdefault(
            "socket_args",
            (address, timeout, source_address),
        ),
    )

    connection = api_client.FixedAddressHTTPSConnection(
        "data.cityofnewyork.us",
        "203.0.113.10",
        timeout=20,
    )
    connection._context = FakeContext()
    connection.connect()

    assert captured["socket_args"] == (("203.0.113.10", 443), 20, None)
    assert captured["server_hostname"] == "data.cityofnewyork.us"


def test_resolve_ipv4_with_dns_over_https_deduplicates_ipv4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_client._RESOLVED_HOST_ADDRESSES.clear()
    captured = {}

    class FakeConnection:
        def __init__(self, host: str, fixed_address: str, timeout: float) -> None:
            captured["init"] = (host, fixed_address, timeout)

        def request(self, method: str, target: str, headers: dict[str, str]) -> None:
            captured["request"] = (method, target, headers)

        def getresponse(self) -> FakeHTTPResponse:
            return FakeHTTPResponse(
                b'{"Answer":[{"type":1,"data":"203.0.113.10"},'
                b'{"type":1,"data":"203.0.113.10"}]}'
            )

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(api_client, "FixedAddressHTTPSConnection", FakeConnection)

    addresses = api_client.resolve_ipv4_with_dns_over_https("example.test")

    assert addresses == ["203.0.113.10"]
    assert captured["init"][0] == api_client.DNS_OVER_HTTPS_SERVER_NAME
    assert captured["init"][1] == api_client.DNS_OVER_HTTPS_ADDRESSES[0]
    assert captured["request"][1] == "/dns-query?name=example.test&type=A"
    assert captured["closed"] is True


def pagination_plan(page_size: int = 2):
    """Return a small deterministic query plan for client pagination tests."""
    return build_query_plan(
        agency="DSNY",
        complaint_type="Graffiti",
        start_date="2024-01-01",
        end_date="2024-01-31",
        selected_columns=["unique_key", "created_date", "agency", "complaint_type"],
        order_by=["created_date ASC", "unique_key ASC"],
        page_size=page_size,
    )


def test_retry_after_is_respected_for_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0, "sleeps": []}

    def fake_urlopen(request: object, timeout: float) -> FakeUrlopenResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            raise HTTPError(
                "https://example.test/resource.json",
                429,
                "Too Many Requests",
                {"Retry-After": "7"},
                io.BytesIO(b"rate limited"),
            )
        return FakeUrlopenResponse(b"[]")

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    monkeypatch.setattr(api_client, "sleep", calls["sleeps"].append)
    response = api_client.fetch_json_response(
        "https://example.test/resource.json",
        timeout_seconds=10,
        max_attempts=2,
        maximum_backoff_seconds=30,
    )
    assert response.payload == []
    assert response.retry_count == 1
    assert calls == {"count": 2, "sleeps": [7.0]}


def test_connection_reset_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_urlopen(request: object, timeout: float) -> FakeUrlopenResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            raise ConnectionResetError("reset")
        return FakeUrlopenResponse(b"[]")

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    monkeypatch.setattr(api_client, "sleep", lambda _: None)
    assert api_client.fetch_json_response(
        "https://example.test/resource.json", timeout_seconds=10, max_attempts=2
    ).retry_count == 1


@pytest.mark.parametrize("status", [400, 401, 403])
def test_non_retryable_http_status_fails_immediately(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    calls = {"count": 0}

    def fake_urlopen(request: object, timeout: float) -> FakeUrlopenResponse:
        calls["count"] += 1
        raise HTTPError(
            "https://example.test/resource.json", status, "Rejected", {}, io.BytesIO(b"bad")
        )

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    with pytest.raises(api_client.APIRequestError, match=f"HTTP {status}"):
        api_client.fetch_json_response(
            "https://example.test/resource.json", timeout_seconds=10, max_attempts=3
        )
    assert calls["count"] == 1


def test_invalid_json_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_urlopen(request: object, timeout: float) -> FakeUrlopenResponse:
        calls["count"] += 1
        return FakeUrlopenResponse(b"invalid")

    monkeypatch.setattr(api_client, "urlopen", fake_urlopen)
    with pytest.raises(api_client.APIRequestError, match="Malformed JSON"):
        api_client.fetch_json_response(
            "https://example.test/resource.json", timeout_seconds=10, max_attempts=3
        )
    assert calls["count"] == 1


def test_paginated_retrieval_uses_offsets_and_short_page_terminates() -> None:
    pages = [
        [
            {"unique_key": "1", "created_date": "2024-01-01T00:00:00.000"},
            {"unique_key": "2", "created_date": "2024-01-02T00:00:00.000"},
        ],
        [{"unique_key": "3", "created_date": "2024-01-03T00:00:00.000"}],
    ]
    offsets: list[int] = []
    captured_headers: list[dict[str, str] | None] = []

    def request(url: str, **kwargs: object) -> api_client.APIResponse:
        query = parse_qs(urlsplit(url).query)["$query"][0]
        offsets.append(int(query.rsplit("OFFSET ", 1)[1]))
        captured_headers.append(kwargs.get("headers"))
        return api_client.APIResponse(pages[len(offsets) - 1], retry_count=0)

    result = api_client.fetch_paginated_records(
        pagination_plan(),
        base_url="https://example.test/resource.json",
        timeout_seconds=10,
        max_attempts=2,
        headers={"X-App-Token": "secret-token"},
        request_json=request,
    )
    assert [row["unique_key"] for row in result.records] == ["1", "2", "3"]
    assert offsets == [0, 2]
    assert [page.returned_rows for page in result.pages] == [2, 1]
    assert captured_headers[0] == {"X-App-Token": "secret-token"}


def test_empty_first_page_returns_audited_empty_result() -> None:
    result = api_client.fetch_paginated_records(
        pagination_plan(),
        base_url="https://example.test/resource.json",
        timeout_seconds=10,
        max_attempts=1,
        request_json=lambda url, **kwargs: api_client.APIResponse([], 0),
    )
    assert result.records == []
    assert len(result.pages) == 1
    assert result.pages[0].offset == 0


@pytest.mark.parametrize("payload", [{"not": "a list"}, ["not an object"]])
def test_unexpected_page_structure_fails(payload: object) -> None:
    with pytest.raises(api_client.APIRequestError, match="list of objects"):
        api_client.fetch_paginated_records(
            pagination_plan(), base_url="https://example.test/resource.json",
            timeout_seconds=10, max_attempts=1,
            request_json=lambda url, **kwargs: api_client.APIResponse(payload, 0),
        )


def test_repeated_page_and_oversized_page_are_rejected() -> None:
    page = [
        {"unique_key": "1", "created_date": "2024-01-01"},
        {"unique_key": "2", "created_date": "2024-01-02"},
    ]
    with pytest.raises(api_client.PaginationIntegrityError, match="Repeated page"):
        api_client.fetch_paginated_records(
            pagination_plan(), base_url="https://example.test/resource.json",
            timeout_seconds=10, max_attempts=1,
            request_json=lambda url, **kwargs: api_client.APIResponse(page, 0),
        )
    with pytest.raises(api_client.PaginationIntegrityError, match="returned 2 rows"):
        api_client.fetch_paginated_records(
            pagination_plan(page_size=1), base_url="https://example.test/resource.json",
            timeout_seconds=10, max_attempts=1,
            request_json=lambda url, **kwargs: api_client.APIResponse(page, 0),
        )


def test_unstable_ordering_is_rejected_and_duplicate_keys_are_preserved() -> None:
    calls = {"count": 0}
    pages = [
        [
            {"unique_key": "same", "created_date": "2024-01-02"},
            {"unique_key": "2", "created_date": "2024-01-03"},
        ],
        [{"unique_key": "same", "created_date": "2024-01-04"}],
    ]

    def duplicate_request(url: str, **kwargs: object) -> api_client.APIResponse:
        value = pages[calls["count"]]
        calls["count"] += 1
        return api_client.APIResponse(value, 0)

    result = api_client.fetch_paginated_records(
        pagination_plan(), base_url="https://example.test/resource.json",
        timeout_seconds=10, max_attempts=1, request_json=duplicate_request,
    )
    assert len(result.records) == 3
    assert result.duplicate_keys_crossing_pages == 1

    bad_page = [
        {"unique_key": "2", "created_date": "2024-01-02"},
        {"unique_key": "1", "created_date": "2024-01-01"},
    ]
    with pytest.raises(api_client.PaginationIntegrityError, match="Unstable"):
        api_client.fetch_paginated_records(
            pagination_plan(), base_url="https://example.test/resource.json",
            timeout_seconds=10, max_attempts=1,
            request_json=lambda url, **kwargs: api_client.APIResponse(bad_page, 0),
        )
