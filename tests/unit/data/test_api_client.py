from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError, URLError
import io
import socket
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from urban_ops.data import api_client


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
