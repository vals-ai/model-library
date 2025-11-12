import io

from PIL import Image

IMAGE_SIZE = 100


def red_image():
    red_img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color="red")
    buffer = io.BytesIO()
    red_img.save(buffer, format="PNG")

    return buffer.getvalue()


def blue_image():
    red_img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color="blue")
    buffer = io.BytesIO()
    red_img.save(buffer, format="PNG")

    return buffer.getvalue()
