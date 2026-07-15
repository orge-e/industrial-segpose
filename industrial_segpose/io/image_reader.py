"""Unicode-safe OpenCV image input."""

from pathlib import Path
import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to read image '{image_path}': {exc}") from exc
    if image is None or image.size == 0:
        raise ValueError(f"OpenCV could not decode image: {image_path}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def discover_images(path: str | Path, recursive: bool = False) -> list[Path]:
    input_path = Path(path)
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def write_image(path: str | Path, image: np.ndarray) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(output_path.suffix or ".png", image)
    if not ok:
        raise OSError(f"Unable to encode output image: {output_path}")
    encoded.tofile(output_path)
