import base64
import io
from collections.abc import Sequence
from typing import cast

from PIL import Image

from model_library.base import FileWithBase64, InputItem


def concat_images(
    images: list[FileWithBase64],
    max_width: int = 1920,
    max_height: int = 1080,
    resize: bool = False,
) -> FileWithBase64:
    """Process multiple base64-encoded images by concatenating them horizontally

    Args:
        images: List of FileWithBase64 objects containing image data
        max_width: Max width of combined image
        max_height: Max height of combined image
        resize: Whether to resize the combined image if it exceeds max dimensions
    """
    # Convert all base64 strings to PIL Images
    pil_images: list[Image.Image] = []
    for image in images:
        image_data = base64.b64decode(image.base64)
        image = Image.open(io.BytesIO(image_data))
        if image.mode != "RGB":
            image = image.convert("RGB")
        pil_images.append(image)

    # Calculate total width and maximum height
    total_width = sum(img.width for img in pil_images)
    max_height = max(img.height for img in pil_images)

    # Create new image with combined dimensions
    combined_image = Image.new("RGB", (total_width, max_height))

    # Paste images horizontally
    x_offset = 0
    for img in pil_images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # Resize if enabled and the image exceeds max dimensions while maintaining aspect ratio
    if resize and (
        combined_image.width > max_width or combined_image.height > max_height
    ):
        # Calculate scaling factor based on both constraints
        width_ratio = max_width / combined_image.width
        height_ratio = max_height / combined_image.height
        scale_factor = min(width_ratio, height_ratio)

        new_width = int(combined_image.width * scale_factor)
        new_height = int(combined_image.height * scale_factor)

        combined_image = combined_image.resize(  # type: ignore
            (new_width, new_height), Image.Resampling.LANCZOS
        )

    # Save to memory buffer for base64 conversion
    buffered = io.BytesIO()
    combined_image.save(buffered, format="JPEG")

    combined_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return FileWithBase64(
        append_type="base64",
        base64=combined_base64,
        mime="jpeg",
        name="combined.jpg",
        type="image",
    )


def trim_images(
    input: Sequence[InputItem],
    max_images: int,
) -> list[InputItem]:
    """Trim images to a maximum count by concatenating excess images.

    Args:
        input: Sequence of input items that may contain images
        max_images: Maximum number of images to allow

    Returns:
        List of input items with images trimmed to max_images count
    """
    input_copy = list(input)
    image_indexes = [
        i for i, item in enumerate(input_copy) if isinstance(item, FileWithBase64)
    ]

    # If more than max_images, concat extras into the last allowed image
    if len(image_indexes) > max_images:
        extra_images = [
            cast(FileWithBase64, input_copy[i]) for i in image_indexes[max_images - 1 :]
        ]
        joined_image = concat_images(
            extra_images,
            max_height=10000,
            max_width=10000,
            resize=True,
        )

        # Replace the last allowed image with the joined one
        input_copy[image_indexes[max_images - 1]] = joined_image

        # Remove the extra images beyond the max
        for i in reversed(image_indexes[max_images:]):
            del input_copy[i]

    return input_copy
