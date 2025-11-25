import os
import shutil
import tarfile
import zipfile
import mimetypes
from PIL import Image
from typing import List, Optional
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
import requests
import base64


os.environ["DOWNLOAD_LATEST_WEIGHTS_MANIFEST"] = "true"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("video/webm", ".webm")

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

IMAGE_TYPES = [".jpg", ".jpeg", ".png", ".webp"]
VIDEO_TYPES = [".mp4", ".mov", ".avi", ".mkv", ".webm"]

with open("examples/api_workflows/birefnet_api.json", "r") as file:
    EXAMPLE_WORKFLOW_JSON = file.read()


class Predictor(BasePredictor):
    def setup(self, weights: str):
        for directory in ALL_DIRECTORIES:
            os.makedirs(directory, exist_ok=True)
        os.makedirs(os.environ.get("YOLO_CONFIG_DIR", "/tmp/Ultralytics"), exist_ok=True)

        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

    def handle_input_file(self, input_file: Path):
        file_extension = self.get_file_extension(input_file)

        if file_extension == ".tar":
            with tarfile.open(input_file, "r") as tar:
                tar.extractall(INPUT_DIR)
        elif file_extension == ".zip":
            with zipfile.ZipFile(input_file, "r") as zip_ref:
                zip_ref.extractall(INPUT_DIR)
        elif file_extension in IMAGE_TYPES + VIDEO_TYPES:
            shutil.copy(input_file, os.path.join(INPUT_DIR, f"input{file_extension}"))
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        print("====================================")
        print(f"Inputs uploaded to {INPUT_DIR}:")
        self.comfyUI.get_files(INPUT_DIR)
        print("====================================")

    def get_file_extension(self, input_file: Path) -> str:
        file_extension = os.path.splitext(input_file)[1].lower()
        if not file_extension:
            with open(input_file, "rb") as f:
                file_signature = f.read(4)
            if file_signature.startswith(b"\x1f\x8b"):  # gzip signature
                file_extension = ".tar"
            elif file_signature.startswith(b"PK"):  # zip signature
                file_extension = ".zip"
            else:
                try:
                    with Image.open(input_file) as img:
                        file_extension = f".{img.format.lower()}"
                        print(f"Determined file type: {file_extension}")
                except Exception as e:
                    raise ValueError(
                        f"Unable to determine file type for: {input_file}, {e}"
                    )
        return file_extension

    def predict(
        self,
        workflow_json: str = Input(
            description="Your ComfyUI workflow as JSON string or URL. You must use the API version of your workflow. Get it from ComfyUI using 'Save (API format)'. Instructions here: https://github.com/replicate/cog-comfyui",
            default="",
        ),
        input_file: Optional[Path] = Input(
            description="Input image, video, tar or zip file. Read guidance on workflows and input files here: https://github.com/replicate/cog-comfyui. Alternatively, you can replace inputs with URLs in your JSON workflow and the model will download them.",
            default=None,
        ),
        return_temp_files: bool = Input(
            description="Return any temporary files, such as preprocessed controlnet images. Useful for debugging.",
            default=False,
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        randomise_seeds: bool = Input(
            description="Automatically randomise seeds (seed, noise_seed, rand_seed)",
            default=True,
        ),
        force_reset_cache: bool = Input(
            description="Force reset the ComfyUI cache before running the workflow. Useful for debugging.",
            default=False,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        if input_file:
            self.handle_input_file(input_file)

        workflow_json_content = workflow_json
        if workflow_json.startswith("data:") and ";base64," in workflow_json:
            try:
                base64_part = workflow_json.split(",", 1)[1]
                decoded_bytes = base64.b64decode(base64_part)
                workflow_json_content = decoded_bytes.decode("utf-8")
            except Exception as e:
                raise ValueError(f"Failed to decode base64 workflow JSON: {e}")
        elif workflow_json.startswith(("http://", "https://")):
            try:
                response = requests.get(workflow_json)
                response.raise_for_status()
                workflow_json_content = response.text
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Failed to download workflow JSON from URL: {e}")

        wf = self.comfyUI.load_workflow(workflow_json_content or EXAMPLE_WORKFLOW_JSON)

        self.comfyUI.connect()

        if force_reset_cache or not randomise_seeds:
            self.comfyUI.reset_execution_cache()

        if randomise_seeds:
            self.comfyUI.randomise_seeds(wf)

        self.comfyUI.run_workflow(wf)

        output_directories = [OUTPUT_DIR]
        if return_temp_files:
            output_directories.append(COMFYUI_TEMP_OUTPUT_DIR)

        optimised_files = optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(output_directories)
        )
        return [Path(p) for p in optimised_files]
