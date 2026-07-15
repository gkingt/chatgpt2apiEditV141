from __future__ import annotations

import json
import os
import unittest
from unittest import mock

import requests

from services.protocol import openai_v1_models


AUTH_KEY = "chatgpt2api"
BASE_URL = "http://localhost:8000"


def model_result(*model_ids: str) -> dict:
    return {
        "object": "list",
        "data": [openai_v1_models._model_entry(model_id) for model_id in model_ids],
    }


class ModelListTests(unittest.TestCase):
    def setUp(self):
        openai_v1_models._clear_model_cache()

    def test_list_models_prefers_authenticated_models_and_adds_latest(self):
        backend = mock.Mock()
        backend.list_models.return_value = model_result("auto", "gpt-5-4-t-mini")

        with (
            mock.patch.object(openai_v1_models, "OpenAIBackendAPI", return_value=backend) as backend_class,
            mock.patch.object(
                openai_v1_models.account_service,
                "get_text_access_token",
                return_value="token-auth",
            ),
            mock.patch.object(openai_v1_models.account_service, "list_accounts", return_value=[]),
        ):
            result = openai_v1_models.list_models()

        ids = [item["id"] for item in result["data"]]
        self.assertEqual(ids[:4], ["auto", "gpt-5-6-sol", "gpt-5-6-Luna", "gpt-5-4-t-mini"])
        backend_class.assert_called_once_with(access_token="token-auth")
        backend.close.assert_called_once_with()

    def test_list_models_rotates_account_after_authenticated_failure(self):
        first_backend = mock.Mock()
        first_backend.list_models.side_effect = RuntimeError("upstream failed")
        second_backend = mock.Mock()
        second_backend.list_models.return_value = model_result("gpt-5-5")

        with (
            mock.patch.object(
                openai_v1_models,
                "OpenAIBackendAPI",
                side_effect=[first_backend, second_backend],
            ) as backend_class,
            mock.patch.object(
                openai_v1_models.account_service,
                "get_text_access_token",
                side_effect=["token-a", "token-b"],
            ) as get_token,
            mock.patch.object(openai_v1_models.account_service, "list_accounts", return_value=[]),
        ):
            result = openai_v1_models.list_models()

        self.assertIn("gpt-5-5", {item["id"] for item in result["data"]})
        self.assertEqual(get_token.call_args_list[1], mock.call({"token-a"}))
        self.assertEqual(
            backend_class.call_args_list,
            [mock.call(access_token="token-a"), mock.call(access_token="token-b")],
        )
        first_backend.close.assert_called_once_with()
        second_backend.close.assert_called_once_with()

    def test_list_models_falls_back_to_anonymous_after_auth_failures(self):
        auth_backends = [mock.Mock() for _ in range(openai_v1_models.MAX_AUTH_MODEL_ATTEMPTS)]
        for backend in auth_backends:
            backend.list_models.side_effect = RuntimeError("auth models failed")
        anonymous_backend = mock.Mock()
        anonymous_backend.list_models.return_value = model_result("auto", "gpt-5-mini")

        with (
            mock.patch.object(
                openai_v1_models,
                "OpenAIBackendAPI",
                side_effect=[*auth_backends, anonymous_backend],
            ) as backend_class,
            mock.patch.object(
                openai_v1_models.account_service,
                "get_text_access_token",
                side_effect=["token-a", "token-b", "token-c"],
            ),
            mock.patch.object(openai_v1_models.account_service, "list_accounts", return_value=[]),
        ):
            result = openai_v1_models.list_models()

        ids = {item["id"] for item in result["data"]}
        self.assertIn("gpt-5-mini", ids)
        self.assertIn("gpt-5-6-sol", ids)
        self.assertIn("gpt-5-6-Luna", ids)
        self.assertEqual(backend_class.call_args_list[-1], mock.call())
        anonymous_backend.close.assert_called_once_with()

    def test_list_models_caches_upstream_result(self):
        backend = mock.Mock()
        backend.list_models.return_value = model_result("auto", "gpt-5-5")

        with (
            mock.patch.object(openai_v1_models, "OpenAIBackendAPI", return_value=backend),
            mock.patch.object(
                openai_v1_models.account_service,
                "get_text_access_token",
                return_value="token-auth",
            ) as get_token,
            mock.patch.object(openai_v1_models.account_service, "list_accounts", return_value=[]),
        ):
            first = openai_v1_models.list_models()
            second = openai_v1_models.list_models()

        self.assertEqual(first, second)
        get_token.assert_called_once_with(set())
        backend.list_models.assert_called_once_with()

    def test_list_models_only_returns_image_models_backed_by_account_types(self):
        backend = mock.Mock()
        backend.list_models.return_value = model_result()

        with (
            mock.patch.object(openai_v1_models, "OpenAIBackendAPI", return_value=backend),
            mock.patch.object(openai_v1_models.account_service, "get_text_access_token", return_value=""),
            mock.patch.object(
                openai_v1_models.account_service,
                "list_accounts",
                return_value=[
                    {"access_token": "token-free", "type": "free"},
                    {"access_token": "token-web-team", "type": "Team", "source_type": "web"},
                    {"access_token": "token-codex-team", "type": "Team", "source_type": "codex"},
                ],
            ),
        ):
            result = openai_v1_models.list_models()

        ids = {item["id"] for item in result["data"]}
        self.assertIn("gpt-image-2", ids)
        self.assertIn("codex-gpt-image-2", ids)
        self.assertIn("team-codex-gpt-image-2", ids)
        self.assertNotIn("plus-codex-gpt-image-2", ids)
        self.assertNotIn("pro-codex-gpt-image-2", ids)

    def test_list_models_does_not_return_codex_models_for_web_plus_accounts(self):
        backend = mock.Mock()
        backend.list_models.return_value = model_result()

        with (
            mock.patch.object(openai_v1_models, "OpenAIBackendAPI", return_value=backend),
            mock.patch.object(openai_v1_models.account_service, "get_text_access_token", return_value=""),
            mock.patch.object(
                openai_v1_models.account_service,
                "list_accounts",
                return_value=[
                    {"access_token": "token-web-plus", "type": "Plus", "source_type": "web"},
                ],
            ),
        ):
            result = openai_v1_models.list_models()

        ids = {item["id"] for item in result["data"]}
        self.assertIn("gpt-image-2", ids)
        self.assertNotIn("codex-gpt-image-2", ids)
        self.assertNotIn("plus-codex-gpt-image-2", ids)


@unittest.skipUnless(os.getenv("RUN_INTEGRATION_TESTS"), "requires a running local API")
class ModelListIntegrationTests(unittest.TestCase):
    def test_list_models_http(self):
        response = requests.get(
            f"{BASE_URL}/v1/models",
            headers={"Authorization": f"Bearer {AUTH_KEY}"},
            timeout=30,
        )
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
