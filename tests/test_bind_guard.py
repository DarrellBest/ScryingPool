"""The address validator. Small, and it is what keeps a second unauthenticated
service off this machine's network — Ollama is already bound 0.0.0.0 here."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from serve.api import DEFAULT_ADDR, check_bind_host, parse_addr  # noqa: E402


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("127.0.0.1:8077", ("127.0.0.1", 8077)),
        ("127.0.0.1:9999", ("127.0.0.1", 9999)),
        ("[::1]:8077", ("::1", 8077)),
        ("::1", ("::1", 8077)),
        ("127.0.0.1", ("127.0.0.1", 8077)),
        (None, ("127.0.0.1", 8077)),
        ("", ("127.0.0.1", 8077)),
    ],
)
def test_loopback_addresses_are_accepted(raw, expected):
    assert parse_addr(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "0.0.0.0:8077",     # the whole reason this function exists
        "[::]:8077",
        "192.168.1.87:8077",
        "100.108.244.43:8077",   # this machine's Tailscale address is still not loopback
        "localhost:8077",        # resolves to loopback, but is not a literal we accept
        "8077",
    ],
)
def test_non_loopback_addresses_are_refused(raw):
    with pytest.raises(ValueError):
        parse_addr(raw)


def test_the_refusal_says_why():
    with pytest.raises(ValueError) as excinfo:
        check_bind_host("0.0.0.0")
    message = str(excinfo.value)
    assert "0.0.0.0" in message
    assert "127.0.0.1" in message
    assert "Tailscale" in message


def test_the_default_address_is_itself_loopback():
    assert parse_addr(DEFAULT_ADDR) == ("127.0.0.1", 8077)


@pytest.mark.parametrize("raw", ["127.0.0.1:not-a-port", "127.0.0.1:0", "127.0.0.1:70000"])
def test_malformed_ports_are_rejected(raw):
    with pytest.raises(ValueError):
        parse_addr(raw)
