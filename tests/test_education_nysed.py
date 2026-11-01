"""NYSED chain completion keeps existing trust roots and verified hostname checks."""

import hashlib
import ssl
from types import SimpleNamespace

import httpx
import pytest

from scripts import education_nysed as n


def client_for(body):
    return httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body)))


def test_tampered_intermediate_fails_before_openssl(monkeypatch):
    monkeypatch.setattr(
        n.subprocess, "run", lambda *a, **kw: pytest.fail("must not verify mismatched pin")
    )
    with client_for(b"tampered") as client, pytest.raises(ValueError, match="pin_mismatch"):
        n.repaired_context(client)


def test_untrusted_issuer_fails_before_context_creation(monkeypatch):
    der = b"untrusted intermediate"
    monkeypatch.setattr(n, "PIN", hashlib.sha256(der).hexdigest())
    monkeypatch.setattr(n.shutil, "which", lambda _: "/usr/bin/openssl")
    monkeypatch.setattr(n.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=2))
    monkeypatch.setattr(
        n.ssl, "create_default_context", lambda **kw: pytest.fail("untrusted cert must not load")
    )
    with client_for(der) as client, pytest.raises(ValueError, match="intermediate_untrusted"):
        n.repaired_context(client)


def test_verified_chain_preserves_root_hostname_and_required_verification(monkeypatch):
    der = b"validated intermediate"
    monkeypatch.setattr(n, "PIN", hashlib.sha256(der).hexdigest())
    monkeypatch.setattr(n.shutil, "which", lambda _: "/usr/bin/openssl")
    commands = []

    def verify(args, **kwargs):
        commands.append((args, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(n.subprocess, "run", verify)

    class Context(ssl.SSLContext):
        def load_verify_locations(self, **kwargs):
            self.added = kwargs

    context = Context(ssl.PROTOCOL_TLS_CLIENT)
    context.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return context

    monkeypatch.setattr(n.ssl, "create_default_context", create)
    with client_for(der) as client:
        result = n.repaired_context(client)
    assert calls == [{"cafile": n.certifi.where()}]
    assert commands[0][0][1:7] == [
        "verify",
        "-CAfile",
        n.certifi.where(),
        "-no-CApath",
        "-no-CAstore",
        commands[0][0][6],
    ]
    assert commands[0][1]["timeout"] == 15
    assert commands[0][1].get("shell", False) is False
    assert result.verify_mode == ssl.CERT_REQUIRED
    assert result.check_hostname is True
    assert not result.verify_flags & ssl.VERIFY_X509_PARTIAL_CHAIN
    assert result.added == {"cadata": ssl.DER_cert_to_PEM_cert(der)}


@pytest.mark.parametrize(
    "url",
    [
        "http://www.nysed.gov/funding-opportunities/test",
        "https://attacker.example/funding-opportunities/test",
        "https://www.nysed.gov@attacker.example/funding-opportunities/test",
    ],
)
def test_source_host_and_tls_are_fixed(url):
    with client_for(b"") as client, pytest.raises(ValueError, match="invalid_url"):
        n._page(client, url)


def test_redirect_not_followed():
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(302, headers={"Location": "https://attacker.example"})
            )
        ) as client,
        pytest.raises(ValueError, match="unavailable"),
    ):
        n._page(client, n.SOURCE)


@pytest.mark.parametrize(
    "body,expected",
    [
        (
            "Submission Deadline & Instructions Documents must be received no later than November 10, 2026. NYSED Designated Contacts",
            True,
        ),
        (
            "Submission Deadline & Instructions Documents must be received no later than October 31, 2026. NYSED Designated Contacts",
            False,
        ),
        ("Questions due November 10, 2026. Contract begins December 1, 2026.", False),
        (
            "Submission Deadline & Instructions Documents must be received November 10, 2026. Bids due November 12, 2026. NYSED Designated Contacts",
            False,
        ),
        (
            "Submission Deadline & Instructions See PDF for submission deadline. RFP Documents",
            False,
        ),
    ],
)
def test_only_unambiguous_future_submission_deadlines(body, expected):
    item = n.qualify("RFP #1 School services", n.SOURCE + "/test", body, "2026-10-31")
    assert bool(item) == expected
    if item:
        assert item["deadline_date"] == "2026-11-10"


def test_higher_education_withheld():
    assert (
        n.qualify(
            "RFP #144 College & University Modernization",
            n.SOURCE,
            "Submission Deadline Documents must be received November 10, 2026. RFP Documents",
            "2026-10-31",
        )
        is None
    )


def test_real_openssl_rejects_self_signed_intermediate(monkeypatch, tmp_path):
    """Even a matching pin cannot introduce an issuer outside existing roots."""
    import subprocess

    executable = n.shutil.which("openssl")
    if not executable:
        pytest.skip("OpenSSL executable unavailable")
    cert = tmp_path / "untrusted.pem"
    subprocess.run(
        [
            executable,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(tmp_path / "key.pem"),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=Untrusted Test Issuer",
        ],
        check=True,
        capture_output=True,
        timeout=15,
    )
    der = ssl.PEM_cert_to_DER_cert(cert.read_text())
    monkeypatch.setattr(n, "PIN", hashlib.sha256(der).hexdigest())
    with client_for(der) as client, pytest.raises(ValueError, match="intermediate_untrusted"):
        n.repaired_context(client)
