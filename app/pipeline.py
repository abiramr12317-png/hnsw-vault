"""
Ingestion pipeline: detect faces -> extract embeddings -> insert into the
HNSW index -> record metadata in the DB. This is what runs when a folder
of event photos gets uploaded.
"""
import os
import shutil
import uuid

from app.embeddings import get_face_embeddings
from app import db


def ingest_image(session, hnsw_index, image_path, storage_dir, display_filename=None):
    """
    Processes a single image end to end. Returns the number of faces found.

    `display_filename` is the original uploaded filename, kept only for
    display purposes. The file itself is stored on disk under a UUID-based
    name so storage never depends on a temp file's name - temp names aren't
    guaranteed to be unique or stable, and using one as a permanent
    identifier is what causes "file not found" errors later.
    """
    os.makedirs(storage_dir, exist_ok=True)

    display_filename = display_filename or os.path.basename(image_path)
    ext = os.path.splitext(display_filename)[1] or os.path.splitext(image_path)[1]
    stored_filename = f"{uuid.uuid4().hex}{ext}"
    stored_path = os.path.join(storage_dir, stored_filename)

    shutil.copy(image_path, stored_path)

    image_id = db.add_image(session, display_filename, stored_path)

    faces = get_face_embeddings(stored_path)
    for embedding, bbox in faces:
        hnsw_id = hnsw_index.insert(embedding)
        db.add_face(session, image_id, hnsw_id, bbox)

    return len(faces)


def ingest_folder(session, hnsw_index, folder_path, storage_dir):
    """Processes every image in a folder. Returns (images_processed, total_faces)."""
    valid_extensions = (".jpg", ".jpeg", ".png")
    images_processed = 0
    total_faces = 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            path = os.path.join(folder_path, filename)
            total_faces += ingest_image(session, hnsw_index, path, storage_dir,
                                         display_filename=filename)
            images_processed += 1

    return images_processed, total_faces