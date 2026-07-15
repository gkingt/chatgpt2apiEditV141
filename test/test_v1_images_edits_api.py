from __future__ import annotations

import base64
import unittest
from io import BytesIO
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import api.ai as ai_module
import services.protocol.openai_v1_image_edit as image_edit_module
from services.protocol.openai_v1_image_edit import _append_mask_instructions, _build_mask_guides, _composite_mask


AUTH_HEADERS = {"Authorization": "Bearer chatgpt2api"}
PNG_BYTES = b"\x89PNG\r\n\x1a\n"
DATA_IMAGE_URL = f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode('ascii')}"
ORIGINAL_IMAGE_EDIT_HANDLE = image_edit_module.handle


class ImagesEditsApiTests(unittest.TestCase):
    def setUp(self):
        self.handle_calls = []

        def fake_handle(payload):
            self.handle_calls.append(payload)
            return {"created": 1, "data": [{"b64_json": base64.b64encode(b"out").decode("ascii")}]}

        self.handler_patcher = mock.patch.object(ai_module.openai_v1_image_edit, "handle", fake_handle)
        self.handler_patcher.start()
        self.addCleanup(self.handler_patcher.stop)
        app = FastAPI()
        app.include_router(ai_module.create_router())
        self.client = TestClient(app)

    def test_edit_accepts_json_image_url(self):
        """测试图片编辑接口支持官方 JSON image_url 引用。"""
        response = self.client.post(
            "/v1/images/edits",
            headers=AUTH_HEADERS,
            json={
                "model": "gpt-image-2",
                "prompt": "edit",
                "images": [{"image_url": DATA_IMAGE_URL}],
                "n": 1,
                "response_format": "b64_json",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(self.handle_calls), 1)
        payload = self.handle_calls[0]
        self.assertEqual(payload["prompt"], "edit")
        self.assertEqual(payload["n"], 1)
        self.assertEqual(payload["images"], [(PNG_BYTES, "image_url.png", "image/png")])

    def test_edit_rejects_file_id_reference(self):
        """测试图片编辑接口对暂不支持的 file_id 返回明确错误。"""
        response = self.client.post(
            "/v1/images/edits",
            headers=AUTH_HEADERS,
            json={
                "model": "gpt-image-2",
                "prompt": "edit",
                "images": [{"file_id": "file-abc123"}],
            },
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("file_id image references are not supported", response.text)
        self.assertEqual(self.handle_calls, [])

    def test_mask_transparent_pixels_mark_the_edit_region(self):
        source_buffer = BytesIO()
        Image.new("RGBA", (2, 1), (20, 40, 60, 255)).save(source_buffer, format="PNG")
        mask = Image.new("L", (2, 1), 255)
        mask.putpixel((0, 0), 0)
        mask_buffer = BytesIO()
        mask.save(mask_buffer, format="PNG")

        result = _composite_mask(
            [(source_buffer.getvalue(), "source.png", "image/png")],
            [(mask_buffer.getvalue(), "mask.png", "image/png")],
        )
        composited = Image.open(BytesIO(result[0][0])).convert("RGBA")

        self.assertEqual(composited.getpixel((0, 0))[3], 0)
        self.assertEqual(composited.getpixel((1, 0))[3], 255)

    def test_mask_creates_visible_green_guide_for_the_edit_region(self):
        source_buffer = BytesIO()
        Image.new("RGBA", (2, 1), (20, 40, 60, 255)).save(source_buffer, format="PNG")
        mask = Image.new("L", (2, 1), 255)
        mask.putpixel((0, 0), 0)
        mask_buffer = BytesIO()
        mask.save(mask_buffer, format="PNG")

        guides = _build_mask_guides(
            [(source_buffer.getvalue(), "source.png", "image/png")],
            [(mask_buffer.getvalue(), "mask.png", "image/png")],
        )
        guide = Image.open(BytesIO(guides[0][0])).convert("RGB")

        self.assertEqual(len(guides), 1)
        self.assertGreater(guide.getpixel((0, 0))[1], guide.getpixel((0, 0))[0])
        self.assertEqual(guide.getpixel((1, 0)), (20, 40, 60))

    def test_opaque_preserve_mask_does_not_create_guide(self):
        source_buffer = BytesIO()
        Image.new("RGB", (1, 1), (20, 40, 60)).save(source_buffer, format="PNG")
        mask_buffer = BytesIO()
        Image.new("L", (1, 1), 255).save(mask_buffer, format="PNG")

        guides = _build_mask_guides(
            [(source_buffer.getvalue(), "source.png", "image/png")],
            [(mask_buffer.getvalue(), "mask.png", "image/png")],
        )

        self.assertEqual(guides, [])

    def test_mask_instructions_explain_visible_and_alpha_regions(self):
        prompt = _append_mask_instructions("replace the plane", 1)

        self.assertIn("附件末尾的 1 张绿色半透明图片", prompt)
        self.assertIn("透明 Alpha", prompt)
        self.assertIn("不要保留绿色蒙版", prompt)

    def test_edit_handle_sends_alpha_source_and_visible_guide(self):
        source_buffer = BytesIO()
        Image.new("RGBA", (2, 1), (20, 40, 60, 255)).save(source_buffer, format="PNG")
        mask = Image.new("L", (2, 1), 255)
        mask.putpixel((0, 0), 0)
        mask_buffer = BytesIO()
        mask.save(mask_buffer, format="PNG")
        captured = {}

        def fake_stream(request):
            captured["request"] = request
            return []

        with (
            mock.patch.object(image_edit_module, "stream_image_outputs_with_pool", side_effect=fake_stream),
            mock.patch.object(image_edit_module, "collect_image_outputs", return_value={"created": 1, "data": []}),
        ):
            ORIGINAL_IMAGE_EDIT_HANDLE({
                "prompt": "replace the plane",
                "images": [(source_buffer.getvalue(), "source.png", "image/png")],
                "mask": [(mask_buffer.getvalue(), "mask.png", "image/png")],
            })

        request = captured["request"]
        self.assertEqual(len(request.images), 2)
        self.assertIn("绿色覆盖区域就是必须修改的区域", request.prompt)

        masked_source = Image.open(BytesIO(base64.b64decode(request.images[0]))).convert("RGBA")
        visible_guide = Image.open(BytesIO(base64.b64decode(request.images[1]))).convert("RGB")
        self.assertEqual(masked_source.getpixel((0, 0))[3], 0)
        self.assertGreater(visible_guide.getpixel((0, 0))[1], visible_guide.getpixel((0, 0))[0])


if __name__ == "__main__":
    unittest.main()
