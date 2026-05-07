import io
from PIL import Image


def pil_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()
