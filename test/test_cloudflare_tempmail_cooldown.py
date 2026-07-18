import unittest
from unittest import mock

from services.register import mail_provider


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.responses.pop(0)

    def close(self):
        pass


class CloudflareTempMailNoCooldownTests(unittest.TestCase):
    def tearDown(self):
        mail_provider.provider_log_sink = None

    @staticmethod
    def provider(session: FakeSession):
        entry = {
            "provider_ref": "cloudflare-test",
            "api_base": "https://mail.example.test",
            "admin_password": "secret",
            "domain": ["example.test"],
        }
        conf = {"request_timeout": 30, "wait_timeout": 30, "wait_interval": 2, "user_agent": "test", "proxy": ""}
        with mock.patch.object(mail_provider, "_create_session", return_value=session):
            return mail_provider.CloudflareTempMailProvider(entry, conf)

    def test_429_fails_current_mailbox_without_sleep_or_retry(self):
        session = FakeSession(
            [
                FakeResponse(429, text="rate limited"),
                FakeResponse(200, {"address": "user@example.test", "jwt": "mail-token"}),
            ]
        )
        provider = self.provider(session)
        logs = []
        mail_provider.provider_log_sink = logs.append
        with mock.patch.object(mail_provider.time, "sleep") as sleep, self.assertRaisesRegex(RuntimeError, "HTTP 429"):
            provider.create_mailbox("user")

        self.assertEqual(len(session.calls), 1)
        sleep.assert_not_called()
        self.assertFalse(logs)

    def test_non_429_error_is_not_retried(self):
        session = FakeSession([FakeResponse(403, text="forbidden")])
        provider = self.provider(session)

        with mock.patch.object(mail_provider.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "HTTP 403"):
                provider.create_mailbox("user")

        self.assertEqual(len(session.calls), 1)
        sleep.assert_not_called()

    def test_wait_for_code_scans_all_list_messages_without_detail_requests(self):
        session = FakeSession(
            [
                FakeResponse(
                    200,
                    {
                        "results": [
                            {"id": "notice", "subject": "Welcome", "text": "No verification code here"},
                            {"id": "otp", "subject": "OpenAI verification", "text": "Verification code: 432198"},
                        ]
                    },
                ),
            ]
        )
        provider = self.provider(session)
        mailbox = {"address": "user@example.test", "token": "mail-token"}

        code = provider.wait_for_code(mailbox)

        self.assertEqual(code, "432198")
        self.assertEqual(len(session.calls), 1)
        self.assertTrue(session.calls[0]["url"].endswith("/api/mails"))

    def test_pre_send_baseline_does_not_consume_just_delivered_code(self):
        session = FakeSession(
            [FakeResponse(200, {"results": [{"id": "otp", "subject": "Verification code: 846210"}]})]
        )
        provider = self.provider(session)
        mailbox = {"address": "user@example.test", "token": "mail-token"}

        provider.prepare_code_baseline(mailbox)
        code = provider.wait_for_code(mailbox)

        self.assertEqual(code, "846210")
        self.assertEqual(len(session.calls), 1)
        self.assertEqual(mailbox.get("_rejected_verification_codes"), [])

    def test_list_message_is_rechecked_when_body_arrives_later(self):
        session = FakeSession(
            [
                FakeResponse(200, {"results": [{"id": "same", "subject": "OpenAI", "text": ""}]}),
                FakeResponse(200, {"results": [{"id": "same", "subject": "OpenAI", "text": "Your code is 654321"}]}),
            ]
        )
        provider = self.provider(session)
        mailbox = {"address": "user@example.test", "token": "mail-token"}

        with mock.patch.object(mail_provider.time, "sleep", return_value=None):
            code = provider.wait_for_code(mailbox)

        self.assertEqual(code, "654321")
        self.assertEqual(len(session.calls), 2)

    def test_account_creation_failed_does_not_add_provider_cooldown_state(self):
        session = FakeSession([FakeResponse(200, {"address": "user@example.test", "jwt": "mail-token"})])
        provider = self.provider(session)
        mailbox = provider.create_mailbox("user")
        logs = []
        mail_provider.provider_log_sink = logs.append

        mail_provider.mark_mailbox_result(
            mailbox,
            success=False,
            error=RuntimeError(
                'user_register_http_400, detail={"error":{"code":"account_creation_failed"}}'
            ),
        )

        self.assertNotIn("_rate_limit_cooldown_key", mailbox)
        self.assertFalse(logs)

    def test_other_registration_errors_do_not_add_provider_cooldown_state(self):
        session = FakeSession([FakeResponse(200, {"address": "user@example.test", "jwt": "mail-token"})])
        provider = self.provider(session)
        mailbox = provider.create_mailbox("user")

        mail_provider.mark_mailbox_result(
            mailbox,
            success=False,
            error=RuntimeError("user_register_http_400, code=invalid_auth_step"),
        )

        self.assertNotIn("_rate_limit_cooldown_key", mailbox)


if __name__ == "__main__":
    unittest.main()
