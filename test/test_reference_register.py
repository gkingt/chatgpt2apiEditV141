from __future__ import annotations

import unittest
from unittest import mock

from services.register import openai_register, reference_register


class ReferenceRegisterTests(unittest.TestCase):
    def test_password_flow_uses_login_hint_and_reference_otp_sender(self) -> None:
        mailbox = {
            "address": "user@example.test",
            "label": "shared-provider",
            "provider": "test",
        }
        registrar = reference_register.ReferencePlatformRegistrar("")
        with (
            mock.patch.object(openai_register, "create_mailbox", return_value=mailbox),
            mock.patch.object(openai_register.mail_provider, "mark_mailbox_result") as mark_result,
            mock.patch.object(registrar, "_chatgpt_authorize") as authorize,
            mock.patch.object(
                registrar,
                "_authorize_signup",
                return_value=("password", ""),
            ) as signup,
            mock.patch.object(registrar, "_register_user") as register_user,
            mock.patch.object(registrar, "_send_email_otp_reference") as send_otp,
            mock.patch.object(registrar, "_validate_mailbox_otp") as validate_otp,
            mock.patch.object(registrar, "_create_account") as create_account,
            mock.patch.object(
                registrar,
                "_finish_chatgpt_registration",
                return_value={"access_token": "chatgpt-token", "session_token": "", "cookie": ""},
            ),
        ):
            result = registrar.register(1)

        registrar.close()
        authorize.assert_called_once_with("user@example.test", 1, include_login_hint=True)
        signup.assert_called_once_with("user@example.test", 1, screen_hint="login_or_signup")
        register_user.assert_called_once()
        send_otp.assert_called_once_with(1, mailbox)
        validate_otp.assert_called_once_with(mailbox, 1)
        create_account.assert_called_once()
        self.assertEqual(result["access_token"], "chatgpt-token")
        self.assertEqual(result["source_type"], "web")
        self.assertEqual(result["registration_engine"], "reference")
        mark_result.assert_called_once_with(mailbox, success=True)

    def test_direct_otp_flow_reuses_shared_resend_and_mailbox_reader(self) -> None:
        mailbox = {"address": "user@example.test", "provider": "test"}
        registrar = reference_register.ReferencePlatformRegistrar("")
        with (
            mock.patch.object(openai_register, "create_mailbox", return_value=mailbox),
            mock.patch.object(openai_register.mail_provider, "mark_mailbox_result"),
            mock.patch.object(registrar, "_chatgpt_authorize"),
            mock.patch.object(
                registrar,
                "_authorize_signup",
                return_value=("otp", "passwordless_signup"),
            ),
            mock.patch.object(registrar, "_register_user") as register_user,
            mock.patch.object(registrar, "_send_email_otp_reference") as send_otp,
            mock.patch.object(registrar, "_resend_signup_otp") as resend,
            mock.patch.object(registrar, "_validate_mailbox_otp"),
            mock.patch.object(registrar, "_create_account"),
            mock.patch.object(
                registrar,
                "_finish_chatgpt_registration",
                return_value={"access_token": "chatgpt-token", "session_token": "", "cookie": ""},
            ),
        ):
            result = registrar.register(2)

        registrar.close()
        register_user.assert_not_called()
        send_otp.assert_not_called()
        resend.assert_called_once_with(2, mailbox)
        self.assertEqual(result["password"], "")

    def test_random_profile_is_used_for_sentinel_generation(self) -> None:
        registrar = reference_register.ReferencePlatformRegistrar("")
        with mock.patch.object(openai_register, "build_sentinel_token", return_value="token") as build:
            self.assertEqual(registrar._build_sentinel_token("authorize_continue"), "token")

        registrar.close()
        call = build.call_args
        self.assertEqual(call.args[1], registrar.device_id)
        self.assertEqual(call.args[2], "authorize_continue")
        self.assertEqual(call.kwargs["user_agent_override"], registrar._browser_user_agent())
        self.assertEqual(call.kwargs["sec_ch_ua_override"], registrar._browser_sec_ch_ua())

    def test_reference_log_context_does_not_replace_main_sink(self) -> None:
        main_logs: list[str] = []
        reference_logs: list[str] = []
        previous = openai_register.register_log_sink
        openai_register.register_log_sink = lambda text, _color="": main_logs.append(text)
        try:
            with openai_register.thread_log_sink(lambda text, _color="": reference_logs.append(text)):
                openai_register.log("reference-only")
            openai_register.log("main-only")
        finally:
            openai_register.register_log_sink = previous

        self.assertEqual(reference_logs, ["reference-only"])
        self.assertEqual(main_logs, ["main-only"])


if __name__ == "__main__":
    unittest.main()
