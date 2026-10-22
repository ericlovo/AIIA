"""Explicit live shadow staging CLI; never called by the offline replay runner.

The caller saves an original envelope before submission. Retries reuse that file
and its UUID. This client cannot approve, promote, fetch sources or invoke models.
"""

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

import httpx

from local_brain.command_center.research_shadow_contract import Brief, _normalized

BASE_URL = "https://managed-leads.vercel.app"
PREFIX = "/api/runner/v1"


class RunnerClientError(ValueError):
    """Intentionally excludes credentials, response bodies and request details."""


def _private_json(path, limit=1024 * 1024):
    path = Path(path).absolute()
    parent = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.getuid()
        or stat.S_IMODE(parent.st_mode) != 0o700
    ):
        raise RunnerClientError("private_parent_required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size > limit
        ):
            raise RunnerClientError("private_file_required")

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise RunnerClientError("invalid_private_json")
                result[key] = value
            return result

        try:
            return json.loads(
                stream.read(limit + 1),
                object_pairs_hook=unique,
                parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
            )
        except (ValueError, UnicodeError):
            raise RunnerClientError("invalid_private_json") from None


def _receipt(body):
    keys = {
        "schemaVersion",
        "receiptId",
        "mode",
        "status",
        "candidateCount",
        "receivedAt",
        "assignmentRevision",
        "payloadHash",
    }
    try:
        if not isinstance(body, dict) or set(body) != keys:
            raise ValueError()
        if (
            type(body["schemaVersion"]) is not int
            or body["schemaVersion"] != 1
            or body["mode"] != "shadow"
            or body["status"] != "shadow_received"
            or type(body["candidateCount"]) is not int
            or not 1 <= body["candidateCount"] <= 500
            or not isinstance(body["receivedAt"], str)
        ):
            raise ValueError()
        UUID(body["receiptId"])
        for key in ("assignmentRevision", "payloadHash"):
            if not re.fullmatch(r"[a-f0-9]{64}", body[key]):
                raise ValueError()
    except (TypeError, ValueError, KeyError, AttributeError):
        raise RunnerClientError("invalid_receipt") from None
    return body


class RunnerClient:
    @classmethod
    def from_file(cls, path, expected_workspace):
        try:
            config = _private_json(path, 16384)
            if not isinstance(config, dict) or set(config) != {
                "baseUrl",
                "workspaceId",
                "runnerId",
                "token",
            }:
                raise ValueError()
            if (
                config["baseUrl"] != BASE_URL
                or config["workspaceId"] != expected_workspace
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", expected_workspace)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", config["runnerId"])
                or not isinstance(config["token"], str)
                or not 32 <= len(config["token"]) <= 512
                or any(not 33 <= ord(c) < 127 for c in config["token"])
            ):
                raise ValueError()
        except (ValueError, TypeError, OSError):
            raise RunnerClientError("invalid_private_credentials") from None
        client = cls()
        client._workspace = expected_workspace
        client._runner = config["runnerId"]
        client._http = httpx.Client(
            base_url=BASE_URL,
            follow_redirects=False,
            timeout=httpx.Timeout(20.0, connect=5.0),
            trust_env=False,
            headers={
                "X-Runner-Client": config["runnerId"],
                "Authorization": "Bearer " + config["token"],
            },
        )
        return client

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _request(self, method, path, payload=None):
        try:
            if self._http.cookies:
                raise RunnerClientError("cookies_forbidden")
            response = self._http.request(method, path, json=payload)
            if "set-cookie" in response.headers:
                self._http.cookies.clear()
                raise RunnerClientError("cookies_forbidden")
            if response.status_code not in (200, 201):
                raise RunnerClientError(f"runner_http_{response.status_code}")
            if len(response.content) > 1024 * 1024:
                raise RunnerClientError("response_too_large")
            return response.status_code, response.json()
        except (httpx.HTTPError, json.JSONDecodeError):
            raise RunnerClientError("runner_transport_failure") from None

    def assignment(self):
        _, body = self._request("GET", PREFIX + "/assignment")
        try:
            if (
                not isinstance(body, dict)
                or set(body)
                != {"schemaVersion", "mode", "workspaceId", "assignmentRevision", "brief"}
                or type(body["schemaVersion"]) is not int
                or body["schemaVersion"] != 1
                or body["mode"] != "shadow"
                or body["workspaceId"] != self._workspace
                or not re.fullmatch(r"[a-f0-9]{64}", body["assignmentRevision"])
            ):
                raise ValueError()
            Brief.model_validate(body["brief"])
        except (ValueError, TypeError, KeyError):
            raise RunnerClientError("invalid_assignment_binding") from None
        return body

    def submit(self, envelope):
        try:
            normalized = _normalized(envelope)
            if len(json.dumps(envelope, allow_nan=False).encode()) > 1024 * 1024:
                raise ValueError()
        except (ValueError, TypeError, KeyError):
            raise RunnerClientError("invalid_submission") from None
        self.assignment()  # Verify actual credential binding, including on exact retries.
        status, body = self._request("POST", PREFIX + "/submissions", envelope)
        if (
            not isinstance(body, dict)
            or set(body) != {"receipt", "replayed"}
            or type(body["replayed"]) is not bool
            or body["replayed"] != (status == 200)
        ):
            raise RunnerClientError("invalid_submission_response")
        receipt = _receipt(body["receipt"])
        expected_hash = hashlib.sha256(
            json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        expected_id = str(
            uuid5(
                NAMESPACE_URL,
                json.dumps(
                    [self._workspace, "runner-shadow", self._runner, normalized["idempotencyKey"]]
                ),
            )
        )
        if (
            receipt["receiptId"] != expected_id
            or receipt["payloadHash"] != expected_hash
            or receipt["assignmentRevision"] != envelope["assignmentRevision"]
            or receipt["candidateCount"] != len(envelope["source"]["candidates"])
        ):
            raise RunnerClientError("receipt_submission_mismatch")
        return status, body

    def receipt(self, receipt_id):
        try:
            if str(UUID(receipt_id)) != receipt_id:
                raise ValueError()
        except (ValueError, TypeError, AttributeError):
            raise RunnerClientError("invalid_receipt_id") from None
        self.assignment()
        receipt = _receipt(self._request("GET", PREFIX + "/submissions/" + receipt_id)[1])
        if receipt["receiptId"] != receipt_id:
            raise RunnerClientError("receipt_id_mismatch")
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("credential-file", "expected-workspace", "envelope-file", "receipt-file"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    try:
        envelope = _private_json(args.envelope_file)
        destination = Path(args.receipt_file).absolute()
        parent = destination.parent.lstat()
        if (
            not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != os.getuid()
            or stat.S_IMODE(parent.st_mode) != 0o700
        ):
            raise RunnerClientError("private_parent_required")
        # Reserve recovery output before sending; no overwrite of earlier receipts.
        fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        with (
            os.fdopen(fd, "w") as output,
            RunnerClient.from_file(args.credential_file, args.expected_workspace) as client,
        ):
            status, body = client.submit(envelope)
            json.dump({"httpStatus": status, **body}, output)
            output.flush()
            os.fsync(output.fileno())
        print("shadow_receipt_saved")
    except (RunnerClientError, OSError):
        print("shadow_submission_failed; preserve original envelope for exact retry")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
