"""Image payload helpers for LLM image inputs and outputs."""

from __future__ import annotations

__all__ = ["LLMImage"]

import base64
from dataclasses import replace
from io import BytesIO
from urllib.request import urlopen
from typing import TYPE_CHECKING, BinaryIO

from heavenbase.utils import exists_file, get_file_ext, load_bin, save_bin
from heavenbase.utils.typing import Any, Dict, Optional, dataclass, field

if TYPE_CHECKING:
    from heavenbase.utils.config import ConfigManager


def _image_url_timeout(config_manager: "ConfigManager | None" = None) -> float:
    if config_manager is None:
        from heavenbase.context import default_context

        config_manager = default_context().config
    manager = config_manager
    default = ((manager.load_default().get("heavenbase", {}) or {}).get("llm", {}) or {}).get("image_url_timeout")
    return float(manager.get("heavenbase.llm.image_url_timeout", default=default))


@dataclass
class LLMImage:
    """Normalize one LLM image source and convert it at provider boundaries.

    Args:
        data (str | None): Base64-encoded image payload.
        format (str): Canonical image format such as `png` or `jpeg`.
        media_type (str): MIME type corresponding to `format`.
        url (str | None): Remote image URL used until bytes are required.
        revised_prompt (str | None): Provider-returned revised generation prompt.
    """

    data: Optional[str] = None
    format: str = "png"
    media_type: str = "image/png"
    url: Optional[str] = None
    revised_prompt: Optional[str] = None
    _url_timeout: Optional[float] = field(default=None, repr=False, compare=False)

    def _bound(self, config_manager: "ConfigManager") -> "LLMImage":
        """Return a copy carrying the selected authority's URL timeout."""
        return replace(self, _url_timeout=_image_url_timeout(config_manager))

    @staticmethod
    def _normalize_format(format: Optional[str] = None) -> str:
        value = (format or "png").strip().lower()
        return "jpeg" if value == "jpg" else value

    @staticmethod
    def _format_from_media_type(media_type: str, fallback: str = "png") -> str:
        token = (media_type or "").strip().lower()
        return LLMImage._normalize_format(token.split("/", 1)[1] if "/" in token else fallback)

    @staticmethod
    def _looks_b64(text: str) -> bool:
        token = text.strip()
        if not token or len(token) % 4 != 0:
            return False
        try:
            base64.b64decode(token, validate=True)
            return True
        except Exception:
            return False

    @classmethod
    def from_bytes(cls, data: bytes, format: str = "png") -> "LLMImage":
        """Create an image from raw bytes.

        Args:
            data (bytes): Encoded image bytes.
            format (str): Source image format. `jpg` normalizes to `jpeg`.

        Returns:
            LLMImage: Base64-backed image value.
        """
        fmt = cls._normalize_format(format)
        return cls(data=base64.b64encode(data).decode("utf-8"), format=fmt, media_type=f"image/{fmt}")

    @classmethod
    def from_pil(cls, image: Any, format: Optional[str] = None) -> "LLMImage":
        """Create an image from a Pillow image object.

        Args:
            image (Any): `PIL.Image.Image` value to encode.
            format (str | None): Output format. Defaults to the image format or `png`.

        Returns:
            LLMImage: Encoded image value.

        Raises:
            ImportError: If Pillow is not installed.
            TypeError: If `image` is not a Pillow image.
        """
        try:
            from PIL import Image as PILImage
        except Exception as exc:
            raise ImportError("Install pillow to normalize PIL image inputs.") from exc
        if not isinstance(image, PILImage.Image):
            raise TypeError(f"PIL image input expected, got {type(image).__name__}")
        fmt = cls._normalize_format(format or image.format or "png")
        out = image.convert("RGB") if fmt == "jpeg" and image.mode in {"RGBA", "LA", "P"} else image
        buffer = BytesIO()
        out.save(buffer, format="JPEG" if fmt == "jpeg" else fmt.upper())
        return cls.from_bytes(buffer.getvalue(), format=fmt)

    @classmethod
    def from_array(cls, array: Any, format: str = "png") -> "LLMImage":
        """Create an image from a NumPy-compatible array.

        Floating values in `[0, 1]` scale to `[0, 255]`; other numeric values
        are clipped before conversion to `uint8`.

        Args:
            array (Any): Array-like image with shape `HxW`, `HxWx1`, `HxWx3`, or `HxWx4`.
            format (str): Output image format.

        Returns:
            LLMImage: Encoded image value.

        Raises:
            ImportError: If NumPy or Pillow is not installed.
            ValueError: If the array shape is not image-compatible.
        """
        try:
            import numpy as np
        except Exception as exc:
            raise ImportError("Install numpy to normalize ndarray image inputs.") from exc
        try:
            from PIL import Image as PILImage
        except Exception as exc:
            raise ImportError("Install pillow to encode ndarray image inputs.") from exc
        data = np.asarray(array)
        if data.ndim == 3 and data.shape[-1] == 1:
            data = data[:, :, 0]
        if data.ndim not in (2, 3) or (data.ndim == 3 and data.shape[-1] not in (3, 4)):
            raise ValueError("ndarray image inputs must have shape HxW, HxWx1, HxWx3, or HxWx4")
        if data.dtype != np.uint8:
            if np.issubdtype(data.dtype, np.floating):
                finite = data[np.isfinite(data)]
                if finite.size and float(finite.min()) >= 0 and float(finite.max()) <= 1:
                    data = data * 255
            data = np.nan_to_num(data, nan=0.0, posinf=255.0, neginf=0.0)
            data = np.clip(data, 0, 255).astype("uint8")
        return cls.from_pil(PILImage.fromarray(data), format=format)

    @classmethod
    def from_file(cls, file: BinaryIO, format: str = "png") -> "LLMImage":
        """Create an image from a readable binary stream.

        Args:
            file (BinaryIO): File-like object whose `read()` returns bytes.
            format (str): Source image format.

        Returns:
            LLMImage: Base64-backed image value.

        Raises:
            TypeError: If the stream does not return bytes or text.
        """
        data = file.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"File-like object must return bytes, got {type(data).__name__}")
        return cls.from_bytes(bytes(data), format=format)

    @classmethod
    def from_path(cls, path: str, format: Optional[str] = None) -> "LLMImage":
        """Create an image from a local file.

        Args:
            path (str): Existing image file path.
            format (str | None): Optional format override. Defaults to the file extension.

        Returns:
            LLMImage: Base64-backed image value.

        Raises:
            FileNotFoundError: If `path` is not an existing file.
        """
        if not exists_file(path):
            raise FileNotFoundError(path)
        fmt = cls._normalize_format(format or get_file_ext(path) or "png")
        return cls.from_bytes(load_bin(path), format=fmt)

    @classmethod
    def from_b64(cls, data: str, format: str = "png") -> "LLMImage":
        """Create an image from base64 data or a data URL.

        Args:
            data (str): Raw base64 payload or `data:image/...;base64,...` URL.
            format (str): Fallback image format when `data` has no MIME type.

        Returns:
            LLMImage: Base64-backed image value.
        """
        media_type = f"image/{cls._normalize_format(format)}"
        value = data
        if value.startswith("data:") and "," in value:
            head, value = value.split(",", 1)
            media_type = head.split(":", 1)[1].split(";", 1)[0]
        fmt = cls._format_from_media_type(media_type, fallback=format)
        return cls(data=value, format=fmt, media_type=f"image/{fmt}")

    @classmethod
    def from_url(cls, url: str, revised_prompt: Optional[str] = None) -> "LLMImage":
        """Create a lazy image value from a URL.

        Args:
            url (str): Remote image URL. It is fetched only when bytes are required.
            revised_prompt (str | None): Optional provider-returned revised prompt.

        Returns:
            LLMImage: URL-backed image value.
        """
        return cls(url=url, revised_prompt=revised_prompt)

    @classmethod
    def from_any(cls, image: Any, format: str = "png") -> "LLMImage":
        """Normalize a supported image source.

        Args:
            image (Any): Existing image, path, URL, data URL, base64 text, bytes-like value,
                stream, Pillow image, NumPy-compatible array, or provider-style mapping.
            format (str): Fallback source/output format.

        Returns:
            LLMImage: Normalized image value.

        Raises:
            TypeError: If the source type or mapping shape is unsupported.
            FileNotFoundError: If a string source is treated as a missing local path.
        """
        if isinstance(image, LLMImage):
            return image
        if isinstance(image, (bytes, bytearray, memoryview)):
            return cls.from_bytes(bytes(image), format=format)
        if isinstance(image, dict):
            url = image.get("url")
            image_url = image.get("image_url")
            if isinstance(image_url, dict):
                url = url or image_url.get("url")
            elif isinstance(image_url, str):
                url = url or image_url
            data = image.get("image") or image.get("data") or image.get("b64_json") or url
            if data is not None:
                return cls.from_any(data, format=format)
            raise TypeError("Image dictionary must contain image, data, b64_json, url, or image_url")
        if isinstance(image, str):
            if image.startswith("data:"):
                return cls.from_b64(image, format=format)
            if image.startswith("http://") or image.startswith("https://"):
                return cls.from_url(image)
            if cls._looks_b64(image):
                return cls.from_b64(image, format=format)
            return cls.from_path(image, format=format)
        if hasattr(image, "read"):
            return cls.from_file(image, format=format)
        if hasattr(image, "save") and hasattr(image, "mode") and hasattr(image, "size"):
            return cls.from_pil(image, format=format)
        if hasattr(image, "shape") and (hasattr(image, "__array__") or hasattr(image, "__array_interface__")):
            return cls.from_array(image, format=format)
        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    @classmethod
    def from_provider_item(cls, item: Any) -> "LLMImage":
        """Create an image from an OpenAI-compatible provider response item.

        It accepts OpenAI `b64_json` and `url` fields plus OpenRouter
        `image_url.url` fields. Data URLs normalize into base64 image payloads.

        Args:
            item (Any): Provider response object or dictionary containing an image reference.

        Returns:
            LLMImage: Normalized provider image value.
        """
        data = item.get("b64_json") if isinstance(item, dict) else getattr(item, "b64_json", None)
        url = item.get("url") if isinstance(item, dict) else getattr(item, "url", None)
        image_url = item.get("image_url") if isinstance(item, dict) else getattr(item, "image_url", None)
        if isinstance(image_url, dict):
            url = url or image_url.get("url")
        elif image_url:
            url = url or getattr(image_url, "url", None)
        image = item.get("image") if isinstance(item, dict) else getattr(item, "image", None)
        url = url or image
        revised_prompt = item.get("revised_prompt") if isinstance(item, dict) else getattr(item, "revised_prompt", None)
        if data:
            if isinstance(data, str) and data.startswith("data:"):
                out = cls.from_b64(data)
                out.revised_prompt = revised_prompt
                return out
            return cls(data=data, url=url, revised_prompt=revised_prompt)
        if isinstance(url, str) and url.startswith("data:"):
            out = cls.from_b64(url)
            out.revised_prompt = revised_prompt
            return out
        return cls(data=data, url=url, revised_prompt=revised_prompt)

    def _ensure_data(self) -> "LLMImage":
        if self.data is not None or not self.url:
            return self
        with urlopen(self.url, timeout=self._url_timeout if self._url_timeout is not None else _image_url_timeout()) as response:
            raw = response.read()
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        self.data = base64.b64encode(raw).decode("utf-8")
        if content_type.startswith("image/"):
            self.media_type = content_type
            self.format = self._format_from_media_type(content_type, fallback=self.format)
        return self

    def to_b64(self) -> str:
        """Return image data as base64.

        Args:
            None.

        Returns:
            str: Base64-encoded image payload.

        Raises:
            ValueError: If neither base64 data nor a URL is available.
        """
        self._ensure_data()
        if self.data is None:
            raise ValueError("Image data is None and no URL is available")
        return self.data

    def to_bytes(self) -> bytes:
        """Return decoded image bytes.

        Args:
            None.

        Returns:
            bytes: Encoded image file bytes.
        """
        return base64.b64decode(self.to_b64())

    def to_data_url(self) -> str:
        """Return image data as a MIME data URL.

        Args:
            None.

        Returns:
            str: `data:` URL containing the base64 image payload.
        """
        return f"data:{self.media_type};base64,{self.to_b64()}"

    def to_dict(self) -> Dict[str, Any]:
        """Return an OpenAI-compatible image content item.

        Args:
            None.

        Returns:
            Dict[str, Any]: `image_url` content item using the remote URL when available.
        """
        return {"type": "image_url", "image_url": {"url": self.url or self.to_data_url()}}

    def to_pil(self) -> Any:
        """Return image data as a Pillow image.

        Args:
            None.

        Returns:
            Any: Open `PIL.Image.Image` backed by the decoded payload.

        Raises:
            ImportError: If Pillow is not installed.
        """
        try:
            from PIL import Image as PILImage
        except Exception as exc:
            raise ImportError("Install pillow to decode image to PIL.") from exc
        return PILImage.open(BytesIO(self.to_bytes()))

    def to_array(self) -> Any:
        """Return image data as a NumPy array.

        Args:
            None.

        Returns:
            Any: NumPy array in Pillow-decoded shape and dtype.

        Raises:
            ImportError: If NumPy or Pillow is not installed.
        """
        try:
            import numpy as np
        except Exception as exc:
            raise ImportError("Install numpy to decode image to ndarray.") from exc
        return np.array(self.to_pil())

    def save(self, path: str) -> None:
        """Save image bytes to a local path.

        Args:
            path (str): Destination file path.

        Returns:
            None: This method does not return a value.
        """
        save_bin(self.to_bytes(), path)
