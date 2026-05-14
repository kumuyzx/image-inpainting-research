#!/usr/bin/env python3
import argparse
import cgi
import json
import logging
import os
import posixpath
import sys
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"
UPLOAD_DIR = ROOT_DIR / "outputs" / "uploads"
RESULT_DIR = ROOT_DIR / "outputs" / "results"
MODEL_DIR = ROOT_DIR / "models" / "big-lama"

sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.evaluation.data import load_image, pad_img_to_modulo
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint

LOGGER = logging.getLogger("lama-web")


class LamaPredictor:
    def __init__(self, model_dir: Path, device: str = "cpu"):
        self.model_dir = model_dir
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def _load_model(self):
        config_path = self.model_dir / "config.yaml"
        checkpoint_path = self.model_dir / "models" / "best.ckpt"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing model config: {config_path}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing model checkpoint: {checkpoint_path}")

        with config_path.open("r", encoding="utf-8") as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = "noop"

        model = load_checkpoint(train_config, str(checkpoint_path), strict=False, map_location="cpu")
        model.freeze()
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_path: Path, mask_path: Path, output_path: Path):
        image = load_image(str(image_path), mode="RGB")
        mask = load_image(str(mask_path), mode="L")[None, ...]
        orig_height, orig_width = image.shape[1:]

        image = pad_img_to_modulo(image, 8)
        mask = pad_img_to_modulo(mask, 8)

        batch = default_collate([{"image": image, "mask": mask}])
        with torch.no_grad():
            batch = move_to_device(batch, self.device)
            batch["mask"] = (batch["mask"] > 0) * 1
            batch = self.model(batch)
            result = batch["inpainted"][0].permute(1, 2, 0).detach().cpu().numpy()

        result = result[:orig_height, :orig_width]
        result = np.clip(result * 255, 0, 255).astype("uint8")
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result)


def _json_response(handler, status, payload):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _safe_name(filename, fallback):
    suffix = Path(filename or "").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"
    return f"{fallback}{suffix}"


class LamaWebHandler(SimpleHTTPRequestHandler):
    predictor = None

    def translate_path(self, path):
        parsed = urlparse(path)
        path = posixpath.normpath(unquote(parsed.path))

        if path.startswith("/results/"):
            rel_path = path[len("/results/"):]
            return str((RESULT_DIR / rel_path).resolve())

        if path in {"/", ""}:
            return str(FRONTEND_DIR / "index.html")

        rel_path = path.lstrip("/")
        return str((FRONTEND_DIR / rel_path).resolve())

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_POST(self):
        if self.path != "/api/inpaint":
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint"})
            return

        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type"),
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                },
            )

            image_item = form["image"] if "image" in form else None
            mask_item = form["mask"] if "mask" in form else None
            if image_item is None or mask_item is None:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Both image and mask files are required."})
                return

            job_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
            job_dir = UPLOAD_DIR / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            image_path = job_dir / _safe_name(image_item.filename, "image")
            mask_path = job_dir / _safe_name(mask_item.filename, "mask")
            output_path = RESULT_DIR / f"{job_id}.png"

            image_path.write_bytes(image_item.file.read())
            mask_path.write_bytes(mask_item.file.read())

            self.predictor.predict(image_path, mask_path, output_path)
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "resultUrl": f"/results/{output_path.name}",
                    "jobId": job_id,
                },
            )
        except Exception as ex:
            LOGGER.error("Prediction failed: %s\n%s", ex, traceback.format_exc())
            _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(ex)})


def main():
    parser = argparse.ArgumentParser(description="Run the LaMa web demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading model from %s", MODEL_DIR)
    LamaWebHandler.predictor = LamaPredictor(MODEL_DIR, device=args.device)
    LOGGER.info("Serving UI from %s", FRONTEND_DIR)
    LOGGER.info("Open http://%s:%s", args.host, args.port)

    server = ThreadingHTTPServer((args.host, args.port), LamaWebHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Server stopped")


if __name__ == "__main__":
    main()
