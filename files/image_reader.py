"""
RENIX Image Reader
==================

Unified image-reading and image-analysis layer for RENIX.

Responsibilities:
- Open common image formats
- Extract image metadata
- Resize/preview images
- Convert images to standard representations
- Extract basic visual properties
- OCR integration
- Pixel/color statistics
- Image validation
- Safe file-size limits

Supported formats depend on Pillow:
PNG, JPG/JPEG, WEBP, BMP, GIF, TIFF, ICO and others.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ============================================================
# EXCEPTIONS
# ============================================================


class ImageReaderError(Exception):
    """Base image-reader exception."""


class ImageDependencyError(ImageReaderError):
    """Raised when Pillow is unavailable."""


class ImageTooLargeError(ImageReaderError):
    """Raised when an image exceeds the configured size limit."""


class InvalidImageError(ImageReaderError):
    """Raised when an image cannot be opened or validated."""


class ImageSecurityError(ImageReaderError):
    """Raised when an image violates safety limits."""


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class ImageMetadata:
    path: str
    filename: str
    extension: str
    format: Optional[str]
    mime_type: Optional[str]
    size_bytes: int
    width: int
    height: int
    mode: str
    has_alpha: bool
    frame_count: int = 1
    animated: bool = False
    dpi: Optional[tuple[float, float]] = None
    color_profile: Optional[str] = None
    hash_sha256: Optional[str] = None
    exif_available: bool = False
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "filename": self.filename,
            "extension": self.extension,
            "format": self.format,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "has_alpha": self.has_alpha,
            "frame_count": self.frame_count,
            "animated": self.animated,
            "dpi": self.dpi,
            "color_profile": self.color_profile,
            "hash_sha256": self.hash_sha256,
            "exif_available": self.exif_available,
            "metadata": self.metadata,
        }


@dataclass
class ImageResult:
    success: bool
    metadata: Optional[ImageMetadata] = None
    image: Any = None
    text: str = ""
    ocr: Optional[dict[str, Any]] = None
    statistics: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    warnings: list[str] = field(
        default_factory=list
    )

    def to_dict(
        self,
        *,
        include_image: bool = False,
    ) -> dict[str, Any]:

        return {
            "success": self.success,
            "metadata": (
                self.metadata.to_dict()
                if self.metadata
                else None
            ),
            "image": (
                self.image
                if include_image
                else None
            ),
            "text": self.text,
            "ocr": self.ocr,
            "statistics": self.statistics,
            "error": self.error,
            "warnings": self.warnings,
        }


# ============================================================
# IMAGE READER
# ============================================================


class ImageReader:
    """
    RENIX image loading and analysis service.

    Pillow is loaded lazily so RENIX can start even if image
    support has not been installed.
    """

    DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024

    SUPPORTED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".gif",
        ".tif",
        ".tiff",
        ".ico",
        ".ppm",
        ".pgm",
        ".pbm",
        ".pnm",
        ".avif",
    }

    MIME_TYPES = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".ico": "image/x-icon",
        ".avif": "image/avif",
    }

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        max_pixels: int = 100_000_000,
        allow_animated: bool = True,
    ) -> None:

        self.enabled = enabled
        self.max_file_size = max_file_size
        self.max_pixels = max_pixels
        self.allow_animated = allow_animated

        logger.info(
            "RENIX ImageReader initialized."
        )

    # ========================================================
    # STATE
    # ========================================================

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def _check_enabled(self) -> None:
        if not self.enabled:
            raise ImageReaderError(
                "ImageReader is disabled."
            )

    # ========================================================
    # DEPENDENCY
    # ========================================================

    @staticmethod
    def _load_pillow():
        try:
            from PIL import Image

            return Image

        except ImportError as exc:

            raise ImageDependencyError(
                "Pillow is required for image support. "
                "Install it with: pip install Pillow"
            ) from exc

    # ========================================================
    # PATH VALIDATION
    # ========================================================

    @staticmethod
    def normalize_path(
        path: str | os.PathLike[str],
    ) -> Path:

        if path is None:
            raise ImageReaderError(
                "Image path cannot be None."
            )

        return Path(path).expanduser()

    def validate_file(
        self,
        path: Path,
    ) -> None:

        if not path.exists():
            raise FileNotFoundError(
                f"Image does not exist: {path}"
            )

        if not path.is_file():
            raise ImageReaderError(
                f"Path is not a file: {path}"
            )

        size = path.stat().st_size

        if size > self.max_file_size:
            raise ImageTooLargeError(
                f"Image size ({size} bytes) exceeds "
                f"limit ({self.max_file_size} bytes)."
            )

    # ========================================================
    # OPEN
    # ========================================================

    def open(
        self,
        path: str | os.PathLike[str],
        *,
        copy: bool = True,
    ):
        """
        Open an image safely.

        Returns:
            PIL.Image.Image
        """

        self._check_enabled()

        image_path = self.normalize_path(path)

        self.validate_file(image_path)

        Image = self._load_pillow()

        try:

            image = Image.open(
                str(image_path)
            )

            # Force basic validation.
            image.verify()

            # Reopen after verify because verify() invalidates
            # the image object.
            image = Image.open(
                str(image_path)
            )

            width, height = image.size

            if width * height > self.max_pixels:

                image.close()

                raise ImageSecurityError(
                    f"Image contains {width * height:,} "
                    f"pixels, exceeding configured limit "
                    f"{self.max_pixels:,}."
                )

            if (
                not self.allow_animated
                and getattr(
                    image,
                    "n_frames",
                    1,
                ) > 1
            ):

                image.close()

                raise ImageSecurityError(
                    "Animated images are disabled."
                )

            if copy:

                result = image.copy()
                image.close()
                return result

            return image

        except ImageSecurityError:
            raise

        except Exception as exc:

            raise InvalidImageError(
                f"Unable to open image: {exc}"
            ) from exc

    # ========================================================
    # HASH
    # ========================================================

    @staticmethod
    def calculate_hash(
        path: Path,
    ) -> str:

        digest = hashlib.sha256()

        with path.open(
            "rb"
        ) as file:

            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):

                digest.update(chunk)

        return digest.hexdigest()

    # ========================================================
    # METADATA
    # ========================================================

    def get_metadata(
        self,
        path: str | os.PathLike[str],
        *,
        calculate_hash: bool = True,
    ) -> ImageMetadata:

        self._check_enabled()

        image_path = self.normalize_path(path)

        self.validate_file(image_path)

        image = self.open(
            image_path,
            copy=False,
        )

        try:

            extension = (
                image_path.suffix.lower()
            )

            image_format = getattr(
                image,
                "format",
                None,
            )

            mode = image.mode

            has_alpha = (
                "A" in mode
                or (
                    image_format == "PNG"
                    and "transparency"
                    in image.info
                )
            )

            frame_count = int(
                getattr(
                    image,
                    "n_frames",
                    1,
                )
            )

            animated = frame_count > 1

            dpi = image.info.get(
                "dpi"
            )

            if dpi is not None:

                try:

                    dpi = (
                        float(dpi[0]),
                        float(dpi[1]),
                    )

                except Exception:
                    dpi = None

            exif = None

            try:
                exif = image.getexif()

            except Exception:
                exif = None

            color_profile = None

            if image.info.get(
                "icc_profile"
            ):
                color_profile = "ICC"

            metadata = ImageMetadata(
                path=str(
                    image_path.resolve()
                ),
                filename=image_path.name,
                extension=extension,
                format=image_format,
                mime_type=self.MIME_TYPES.get(
                    extension
                ),
                size_bytes=image_path.stat().st_size,
                width=image.width,
                height=image.height,
                mode=mode,
                has_alpha=has_alpha,
                frame_count=frame_count,
                animated=animated,
                dpi=dpi,
                color_profile=color_profile,
                hash_sha256=(
                    self.calculate_hash(
                        image_path
                    )
                    if calculate_hash
                    else None
                ),
                exif_available=bool(exif),
                metadata={
                    "info_keys": list(
                        image.info.keys()
                    ),
                },
            )

            return metadata

        finally:

            image.close()

    # ========================================================
    # READ
    # ========================================================

    def read(
        self,
        path: str | os.PathLike[str],
        *,
        load_image: bool = True,
        calculate_hash: bool = True,
    ) -> ImageResult:

        self._check_enabled()

        image_path = self.normalize_path(path)

        try:

            metadata = self.get_metadata(
                image_path,
                calculate_hash=calculate_hash,
            )

            image = None

            if load_image:

                image = self.open(
                    image_path,
                    copy=True,
                )

            return ImageResult(
                success=True,
                metadata=metadata,
                image=image,
            )

        except Exception as exc:

            logger.exception(
                "Failed to read image."
            )

            return ImageResult(
                success=False,
                error=str(exc),
            )

    # ========================================================
    # RGB CONVERSION
    # ========================================================

    def to_rgb(
        self,
        image: Any,
    ):
        Image = self._load_pillow()

        if image.mode == "RGB":
            return image.copy()

        return image.convert(
            "RGB"
        )

    def to_rgba(
        self,
        image: Any,
    ):
        Image = self._load_pillow()

        if image.mode == "RGBA":
            return image.copy()

        return image.convert(
            "RGBA"
        )

    # ========================================================
    # RESIZE
    # ========================================================

    def resize(
        self,
        image: Any,
        width: int,
        height: Optional[int] = None,
        *,
        keep_aspect: bool = True,
    ):
        """
        Resize a PIL image.

        If height is omitted and keep_aspect is True,
        height is calculated automatically.
        """

        if width <= 0:
            raise ValueError(
                "width must be positive."
            )

        Image = self._load_pillow()

        original_width, original_height = (
            image.size
        )

        if keep_aspect:

            if height is None:

                ratio = (
                    width
                    / original_width
                )

                height = max(
                    1,
                    int(
                        original_height
                        * ratio
                    ),
                )

            else:

                ratio = min(
                    width / original_width,
                    height / original_height,
                )

                width = max(
                    1,
                    int(
                        original_width
                        * ratio
                    ),
                )

                height = max(
                    1,
                    int(
                        original_height
                        * ratio
                    ),
                )

        elif height is None:

            raise ValueError(
                "height is required when "
                "keep_aspect=False."
            )

        resampling = getattr(
            Image,
            "Resampling",
            Image,
        )

        return image.resize(
            (width, height),
            resampling.LANCZOS,
        )

    # ========================================================
    # THUMBNAIL
    # ========================================================

    def thumbnail(
        self,
        image: Any,
        max_width: int = 512,
        max_height: int = 512,
    ):
        """
        Create a copy constrained to a bounding box.
        """

        if (
            max_width <= 0
            or max_height <= 0
        ):
            raise ValueError(
                "Thumbnail dimensions must be positive."
            )

        result = image.copy()

        result.thumbnail(
            (
                max_width,
                max_height,
            )
        )

        return result

    # ========================================================
    # BYTES
    # ========================================================

    def to_bytes(
        self,
        image: Any,
        *,
        format: str = "PNG",
        quality: int = 90,
    ) -> bytes:

        buffer = io.BytesIO()

        save_kwargs: dict[str, Any] = {}

        if format.upper() in {
            "JPEG",
            "JPG",
        }:

            save_kwargs["quality"] = max(
                1,
                min(100, quality),
            )

            if image.mode in {
                "RGBA",
                "LA",
                "P",
            }:

                image = image.convert(
                    "RGB"
                )

        image.save(
            buffer,
            format=format,
            **save_kwargs,
        )

        return buffer.getvalue()

    # ========================================================
    # IMAGE STATISTICS
    # ========================================================

    def statistics(
        self,
        image: Any,
    ) -> dict[str, Any]:

        ImageStat = None

        try:

            from PIL import ImageStat

        except ImportError as exc:

            raise ImageDependencyError(
                "Pillow is required."
            ) from exc

        rgb = self.to_rgb(image)

        try:

            stat = ImageStat.Stat(rgb)

            means = stat.mean
            extrema = stat.extrema

            return {
                "mode": image.mode,
                "width": image.width,
                "height": image.height,
                "pixels": (
                    image.width
                    * image.height
                ),
                "mean_rgb": [
                    round(value, 3)
                    for value in means
                ],
                "extrema_rgb": extrema,
            }

        finally:

            rgb.close()

    # ========================================================
    # COLORS
    # ========================================================

    def dominant_colors(
        self,
        image: Any,
        *,
        count: int = 5,
    ) -> list[tuple[int, int, int]]:

        if count <= 0:
            raise ValueError(
                "count must be positive."
            )

        rgb = self.to_rgb(image)

        try:

            # Quantization gives a compact palette.
            quantized = rgb.quantize(
                colors=count
            )

            palette = (
                quantized.getpalette()
            )

            color_counts = (
                quantized.getcolors()
            )

            if not color_counts:
                return []

            color_counts.sort(
                reverse=True
            )

            colors = []

            for _, palette_index in (
                color_counts[:count]
            ):

                offset = (
                    palette_index * 3
                )

                if (
                    offset + 2
                    >= len(palette)
                ):
                    continue

                colors.append(
                    (
                        palette[offset],
                        palette[offset + 1],
                        palette[offset + 2],
                    )
                )

            return colors

        finally:

            rgb.close()

    # ========================================================
    # OCR
    # ========================================================

    def ocr(
        self,
        image: Any,
        *,
        language: str = "eng",
    ) -> dict[str, Any]:

        """
        OCR adapter.

        Priority:
        1. pytesseract
        2. informative error

        Tesseract itself must be installed separately.
        """

        try:

            import pytesseract

        except ImportError:

            return {
                "success": False,
                "text": "",
                "error": (
                    "pytesseract is not installed. "
                    "Install with: pip install pytesseract"
                ),
            }

        try:

            text = pytesseract.image_to_string(
                image,
                lang=language,
            )

            return {
                "success": True,
                "text": text,
                "language": language,
            }

        except Exception as exc:

            return {
                "success": False,
                "text": "",
                "error": str(exc),
            }

    # ========================================================
    # IMAGE + OCR
    # ========================================================

    def read_with_ocr(
        self,
        path: str | os.PathLike[str],
        *,
        language: str = "eng",
    ) -> ImageResult:

        result = self.read(path)

        if not result.success:
            return result

        if result.image is None:

            return ImageResult(
                success=False,
                metadata=result.metadata,
                error="Image was not loaded.",
            )

        try:

            ocr_result = self.ocr(
                result.image,
                language=language,
            )

            result.ocr = ocr_result
            result.text = (
                ocr_result.get(
                    "text",
                    "",
                )
            )

            return result

        finally:

            try:
                result.image.close()

            except Exception:
                pass

    # ========================================================
    # PREVIEW
    # ========================================================

    def create_preview(
        self,
        path: str | os.PathLike[str],
        *,
        max_width: int = 1024,
        max_height: int = 1024,
        output_format: str = "PNG",
    ) -> bytes:

        result = self.read(path)

        if not result.success:

            raise ImageReaderError(
                result.error
                or "Unable to read image."
            )

        if result.image is None:

            raise ImageReaderError(
                "Image was not loaded."
            )

        try:

            preview = self.thumbnail(
                result.image,
                max_width,
                max_height,
            )

            try:

                return self.to_bytes(
                    preview,
                    format=output_format,
                )

            finally:

                preview.close()

        finally:

            result.image.close()

    # ========================================================
    # FRAME EXTRACTION
    # ========================================================

    def get_frame(
        self,
        path: str | os.PathLike[str],
        frame_number: int = 0,
    ):
        """
        Extract a frame from an animated image.
        """

        if frame_number < 0:
            raise ValueError(
                "frame_number must be >= 0."
            )

        image = self.open(
            path,
            copy=False,
        )

        try:

            frame_count = int(
                getattr(
                    image,
                    "n_frames",
                    1,
                )
            )

            if frame_number >= frame_count:

                raise IndexError(
                    f"Image has {frame_count} frames."
                )

            image.seek(frame_number)

            return image.convert(
                "RGBA"
            ).copy()

        finally:

            image.close()

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(
        self,
        path: str | os.PathLike[str],
    ) -> dict[str, Any]:

        try:

            metadata = self.get_metadata(
                path
            )

            return {
                "valid": True,
                "metadata": metadata.to_dict(),
                "error": None,
            }

        except Exception as exc:

            return {
                "valid": False,
                "metadata": None,
                "error": str(exc),
            }

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self) -> dict[str, Any]:

        pillow_available = True

        try:
            self._load_pillow()

        except ImageDependencyError:
            pillow_available = False

        return {
            "enabled": self.enabled,
            "pillow_available": pillow_available,
            "max_file_size": self.max_file_size,
            "max_pixels": self.max_pixels,
            "allow_animated": self.allow_animated,
            "supported_extensions": sorted(
                self.SUPPORTED_EXTENSIONS
            ),
            "capabilities": [
                "image_loading",
                "metadata",
                "validation",
                "sha256_hash",
                "rgb_conversion",
                "rgba_conversion",
                "resize",
                "thumbnail",
                "bytes_conversion",
                "statistics",
                "dominant_colors",
                "ocr",
                "preview",
                "animated_frame_extraction",
            ],
        }

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def shutdown(self) -> None:

        self.enabled = False

        logger.info(
            "RENIX ImageReader shut down."
        )


# ============================================================
# DEFAULT INSTANCE
# ============================================================

_default_reader: Optional[
    ImageReader
] = None


def get_image_reader() -> ImageReader:

    global _default_reader

    if _default_reader is None:
        _default_reader = ImageReader()

    return _default_reader


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================


def read_image(
    path: str | os.PathLike[str],
) -> ImageResult:

    return get_image_reader().read(path)


def image_metadata(
    path: str | os.PathLike[str],
) -> ImageMetadata:

    return get_image_reader().get_metadata(
        path
    )


def validate_image(
    path: str | os.PathLike[str],
) -> dict[str, Any]:

    return get_image_reader().validate(
        path
    )


def image_ocr(
    path: str | os.PathLike[str],
    *,
    language: str = "eng",
) -> ImageResult:

    return get_image_reader().read_with_ocr(
        path,
        language=language,
    )


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "ImageReaderError",
    "ImageDependencyError",
    "ImageTooLargeError",
    "InvalidImageError",
    "ImageSecurityError",
    "ImageMetadata",
    "ImageResult",
    "ImageReader",
    "get_image_reader",
    "read_image",
    "image_metadata",
    "validate_image",
    "image_ocr",
]


