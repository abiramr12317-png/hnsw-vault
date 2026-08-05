"""
Lazy thumbnail generation, with on-disk caching.

Thumbnails are NOT generated at upload time - only when an image is actually
about to be shown to the user (e.g. as a search result). Out of 1000+ uploaded
photos, a typical search only returns a handful of matches, so generating
thumbnails for everything upfront wastes work on images that may never be
viewed. Once a thumbnail is generated it's cached on disk, so it's only ever
computed once per image.
"""
import os
from PIL import Image as PILImage

THUMB_SIZE = (256, 256)


def get_thumbnail_path(original_path, thumb_dir):
    filename = os.path.basename(original_path)
    return os.path.join(thumb_dir, filename)


def get_or_create_thumbnail(original_path, thumb_dir):
    """Returns the path to a thumbnail for the given image, generating it if needed."""
    os.makedirs(thumb_dir, exist_ok=True)
    thumb_path = get_thumbnail_path(original_path, thumb_dir)

    if os.path.exists(thumb_path):
        return thumb_path  # cache hit - nothing to do

    with PILImage.open(original_path) as img:
        img = img.convert("RGB")
        img.thumbnail(THUMB_SIZE)
        img.save(thumb_path, "JPEG", quality=80)

    return thumb_path