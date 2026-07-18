from __future__ import annotations

import re
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from services.register import mail_provider


class FakeResponse:
    def __init__(self, status_code: int, payload: object, text: str = ""):
        self.status_code = status_code
        self.payload = payload
        self.text = text

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse | Exception]):
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []
        self.headers: dict[str, str] = {}
        self.closed = False

    def request(self, method: str, url: str, **kwargs):
        self.requests.append({"method": method, "url": url, **kwargs})
        if not self.responses:
            raise AssertionError("unexpected request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        self.closed = True


class TempMailLolProviderTests(unittest.TestCase):
    conf = {
        "request_timeout": 30.0,
        "wait_timeout": 2.0,
        "wait_interval": 0.01,
        "user_agent": "test-agent",
        "proxy": "",
    }

    def make_provider(self, entry: dict[str, object], session: FakeSession) -> mail_provider.TempMailLolProvider:
        with mock.patch.object(mail_provider, "_create_session", return_value=session):
            return mail_provider.TempMailLolProvider(entry, dict(self.conf))

    def test_parses_multiple_keys(self) -> None:
        self.assertEqual(mail_provider._parse_tempmail_keys(" key-a, key-b\nkey-a "), ["key-a", "key-b"])
        self.assertEqual(mail_provider._parse_tempmail_keys(""), [""])

    def test_parses_optional_multiline_domains(self) -> None:
        self.assertEqual(
            mail_provider.parse_tempmail_domains(" @First.Example\nsecond.example,first.example. "),
            ["first.example", "second.example"],
        )
        self.assertEqual(mail_provider.parse_tempmail_domains(""), [])

    def test_create_429_fails_fast_without_key_rotation_or_sleep(self) -> None:
        session = FakeSession(
            [
                FakeResponse(429, {"error": "rate limited"}, "rate limited"),
                FakeResponse(201, {"address": "user@abc12.example.com", "token": "token-429"}),
            ]
        )
        provider = self.make_provider(
            {
                "provider_ref": "tempmail-test-429",
                "api_key": "key-one\nkey-two",
                "domain": ["abc12.example.com"],
            },
            session,
        )
        logs: list[str] = []
        mail_provider.provider_log_sink = logs.append
        try:
            with mock.patch.object(mail_provider.time, "sleep") as sleep, self.assertRaisesRegex(
                RuntimeError, "HTTP 429"
            ):
                provider.create_mailbox()
        finally:
            mail_provider.provider_log_sink = None

        sleep.assert_not_called()
        self.assertFalse(logs)
        self.assertNotIn("key-one", "\n".join(logs))
        self.assertNotIn("token-429", "\n".join(logs))
        self.assertEqual([request["headers"] for request in session.requests], [{"Authorization": "Bearer key-one"}])
        payload = session.requests[0]["json"]
        self.assertIsInstance(payload, dict)
        assert isinstance(payload, dict)
        self.assertEqual(payload["domain"], "abc12.example.com")
        self.assertTrue(re.fullmatch(r"[a-z]{5}\d{1,3}[a-z]{1,3}", str(payload["prefix"])))

    def test_key_rotation_is_shared_without_cooldown(self) -> None:
        entry = {
            "provider_ref": "tempmail-shared-429",
            "api_key": "key-one\nkey-two",
        }
        first = self.make_provider(entry, FakeSession([]))
        second = self.make_provider(entry, FakeSession([]))

        self.assertIs(first.key_pool, second.key_pool)
        self.assertEqual(first.key_pool.next_key(), "key-one")
        self.assertEqual(second.key_pool.next_key(), "key-two")
        self.assertEqual(first.key_pool.next_key(), "key-one")

    def test_create_fails_fast_for_fatal_4xx(self) -> None:
        session = FakeSession([FakeResponse(403, {"error": "forbidden"}, "forbidden")])
        provider = self.make_provider(
            {
                "provider_ref": "tempmail-test-fatal",
                "api_key": "bad-key\nunused-key",
                "domain": [],
                "max_wait": 0,
            },
            session,
        )

        with self.assertRaisesRegex(RuntimeError, r"创建邮箱失败 \(HTTP 403\)"):
            provider.create_mailbox()
        self.assertEqual(len(session.requests), 1)

    def test_create_network_error_fails_current_task_without_retry(self) -> None:
        session = FakeSession(
            [
                OSError("temporary disconnect"),
                FakeResponse(201, {"address": "mail@example.com", "token": "token-network"}),
            ]
        )
        provider = self.make_provider(
            {
                "provider_ref": "tempmail-test-network",
                "api_key": "key-one,key-two",
                "domain": [],
                "max_wait": 0,
            },
            session,
        )

        with mock.patch.object(mail_provider.time, "sleep") as sleep, self.assertRaisesRegex(
            RuntimeError, "network"
        ):
            provider.create_mailbox()

        sleep.assert_not_called()
        self.assertEqual(
            [request["headers"] for request in session.requests],
            [{"Authorization": "Bearer key-one"}],
        )

    def test_configured_domain_is_sent_and_created_address_is_accepted(self) -> None:
        session = FakeSession([FakeResponse(201, {"address": "mail@random-provider.example", "token": "token"})])
        provider = self.make_provider(
            {
                "provider_ref": "tempmail-test-domain",
                "api_key": "key",
                "domain": ["requested.example"],
            },
            session,
        )

        mailbox = provider.create_mailbox()

        self.assertEqual(mailbox["address"], "mail@random-provider.example")
        self.assertEqual(len(session.requests), 1)
        self.assertEqual(session.requests[0]["json"]["domain"], "requested.example")

    def test_empty_domain_is_not_sent(self) -> None:
        session = FakeSession([FakeResponse(201, {"address": "mail@automatic.example", "token": "token-auto"})])
        provider = self.make_provider(
            {
                "provider_ref": "tempmail-test-auto-domain",
                "api_key": "key",
                "domain": ["", "  "],
            },
            session,
        )

        provider.create_mailbox()

        self.assertNotIn("domain", session.requests[0]["json"])

    def test_multiple_domains_rotate_across_provider_instances(self) -> None:
        first_session = FakeSession(
            [FakeResponse(201, {"address": "first@one.example", "token": "token-one"})]
        )
        second_session = FakeSession(
            [FakeResponse(201, {"address": "second@two.example", "token": "token-two"})]
        )
        entry = {
            "provider_ref": "tempmail-test-domain-rotation",
            "api_key": "key",
            "domain": "one.example\ntwo.example",
        }
        first_provider = self.make_provider(entry, first_session)
        second_provider = self.make_provider(entry, second_session)

        first_provider.create_mailbox()
        second_provider.create_mailbox()

        self.assertEqual(
            [
                first_session.requests[0]["json"]["domain"],
                second_session.requests[0]["json"]["domain"],
            ],
            ["one.example", "two.example"],
        )

    def test_key_pool_round_robins_without_sliding_window_limit(self) -> None:
        pool = mail_provider._TempMailKeyPool(["key-one", "key-two"])

        self.assertEqual(pool.next_key(), "key-one")
        self.assertEqual(pool.next_key(), "key-two")
        self.assertEqual(pool.next_key(), "key-one")

    def test_poll_uses_creation_key_then_switches_after_three_errors(self) -> None:
        entry = {
            "provider_ref": "tempmail-test-poll",
            "api_key": "primary-key\nfallback-key",
            "domain": [],
            "max_wait": 0,
        }
        create_session = FakeSession([FakeResponse(201, {"address": "mail@example.com", "token": "token-poll"})])
        creator = self.make_provider(entry, create_session)
        mailbox = creator.create_mailbox()
        mailbox["_code_not_before"] = datetime.now(timezone.utc)

        poll_session = FakeSession(
            [
                FakeResponse(500, {}, "temporary"),
                FakeResponse(503, {}, "temporary"),
                FakeResponse(520, {}, "temporary"),
                FakeResponse(
                    200,
                    {
                        "emails": [
                            {"id": "newest", "subject": "Status update", "body": "No code here"},
                            {"id": "code", "subject": "Your verification code is 432198", "body": "Use it now"},
                        ]
                    },
                ),
            ]
        )
        poller = self.make_provider(entry, poll_session)

        with mock.patch.object(mail_provider.time, "sleep", return_value=None):
            code = poller.wait_for_code(mailbox)

        self.assertEqual(code, "432198")
        self.assertEqual(
            [request["headers"] for request in poll_session.requests],
            [
                {"Authorization": "Bearer primary-key"},
                {"Authorization": "Bearer primary-key"},
                {"Authorization": "Bearer primary-key"},
                {"Authorization": "Bearer fallback-key"},
            ],
        )

    def test_poll_rechecks_message_when_body_arrives_later(self) -> None:
        entry = {"provider_ref": "tempmail-recheck", "api_key": "key", "domain": []}
        session = FakeSession(
            [
                FakeResponse(200, {"emails": [{"id": "same-message", "subject": "OpenAI", "body": ""}]}),
                FakeResponse(
                    200,
                    {"emails": [{"id": "same-message", "subject": "OpenAI", "body": "Your ChatGPT code is 654321"}]},
                ),
            ]
        )
        provider = self.make_provider(entry, session)
        mailbox = {"address": "mail@example.com", "token": "token-recheck"}

        with mock.patch.object(mail_provider.time, "sleep", return_value=None):
            code = provider.wait_for_code(mailbox)

        self.assertEqual(code, "654321")
        self.assertEqual(len(session.requests), 2)

    def test_pre_send_baseline_does_not_consume_just_delivered_code(self) -> None:
        entry = {"provider_ref": "tempmail-baseline", "api_key": "key", "domain": []}
        session = FakeSession(
            [FakeResponse(200, {"emails": [{"id": "new-message", "subject": "Verification code: 846210"}]})]
        )
        provider = self.make_provider(entry, session)
        mailbox = {"address": "mail@example.com", "token": "token-baseline"}

        provider.prepare_code_baseline(mailbox)
        code = provider.wait_for_code(mailbox)

        self.assertEqual(code, "846210")
        self.assertEqual(len(session.requests), 1)
        self.assertEqual(mailbox.get("_rejected_verification_codes"), [])

    def test_poll_accepts_unseen_message_id_despite_clock_skew(self) -> None:
        entry = {"provider_ref": "tempmail-clock-skew", "api_key": "key", "domain": []}
        session = FakeSession(
            [
                FakeResponse(
                    200,
                    {
                        "emails": [
                            {
                                "id": "new-message",
                                "subject": "Verification code: 321654",
                                "created_at": "2020-01-01T00:00:00Z",
                            }
                        ]
                    },
                )
            ]
        )
        provider = self.make_provider(entry, session)
        mailbox = {
            "address": "mail@example.com",
            "token": "token-clock-skew",
            "_code_not_before": datetime.now(timezone.utc),
        }

        self.assertEqual(provider.wait_for_code(mailbox), "321654")

    def test_poll_429_fails_fast_without_leaking_secrets(self) -> None:
        entry = {
            "provider_ref": "tempmail-poll-429",
            "api_key": "api-secret",
            "domain": [],
        }
        session = FakeSession([FakeResponse(429, {"error": "rate limited"}, "rate limited")])
        provider = self.make_provider(entry, session)
        mailbox = {"address": "mail@example.com", "token": "token-secret"}
        logs: list[str] = []
        mail_provider.provider_log_sink = logs.append
        try:
            with mock.patch.object(mail_provider.time, "sleep") as sleep, self.assertRaisesRegex(
                RuntimeError, "HTTP 429"
            ):
                provider.wait_for_code(mailbox)
        finally:
            mail_provider.provider_log_sink = None

        self.assertEqual(len(session.requests), 1)
        sleep.assert_not_called()
        summary = "\n".join(logs)
        self.assertIn("http_429=1", summary)
        self.assertNotIn("cooldown", summary)
        self.assertNotIn("api-secret", summary)
        self.assertNotIn("token-secret", summary)

    def test_domain_history_never_skips_created_address(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            mail_provider, "TEMPMAIL_DOMAIN_STATS_FILE", Path(temp_dir) / "domain-stats.json"
        ):
            for _ in range(3):
                mail_provider._record_tempmail_domain_result(
                    "abc.airfryersbg.com",
                    received=False,
                )
            session = FakeSession(
                [
                    FakeResponse(201, {"address": "direct@next.airfryersbg.com", "token": "accepted-token"}),
                ]
            )
            provider = self.make_provider(
                {
                    "provider_ref": "tempmail-domain-cooldown",
                    "api_key": "key",
                    "domain": ["whitelist.example"],
                    "max_wait": 0,
                    "domain_cooldown_threshold": 3,
                    "domain_cooldown_seconds": 600,
                },
                session,
            )

            mailbox = provider.create_mailbox()
            stats = {item["domain"]: item for item in mail_provider.tempmail_domain_stats_snapshot()}

        self.assertEqual(mailbox["address"], "direct@next.airfryersbg.com")
        self.assertEqual(len(session.requests), 1)
        self.assertEqual(session.requests[0]["json"]["domain"], "whitelist.example")
        self.assertEqual(stats["airfryersbg.com"]["consecutive_timeouts"], 3)
        self.assertNotIn("cooling", stats["airfryersbg.com"])

    def test_delivery_result_is_recorded_only_once_per_mailbox(self) -> None:
        mailbox = {
            "provider": "tempmail_lol",
            "address": "mail@tal.gardianwaves.org",
        }
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            mail_provider, "TEMPMAIL_DOMAIN_STATS_FILE", Path(temp_dir) / "domain-stats.json"
        ):
            mail_provider.mark_verification_code_received(mailbox)
            mail_provider.mark_verification_code_received(mailbox)
            mail_provider.mark_mailbox_result(mailbox, success=False, error="等待注册验证码超时")
            stats = {item["domain"]: item for item in mail_provider.tempmail_domain_stats_snapshot()}

        self.assertEqual(stats["gardianwaves.org"]["received"], 1)
        self.assertEqual(stats["gardianwaves.org"]["timeouts"], 0)


if __name__ == "__main__":
    unittest.main()
