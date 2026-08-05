"""
FastAPI app for HNSW Vault.

Two endpoints:
  POST /upload  - bulk-ingest photos (runs the ingestion pipeline)
  POST /search  - upload a query face, get back matching photos

Run with:  uvicorn app.main:app --reload
"""
import os
import shutil
import tempfile

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app import config, db
from app.hnsw import HNSWIndex
from app.pipeline import ingest_image
from app.embeddings import get_face_embeddings
from app.thumbnails import get_or_create_thumbnail

app = FastAPI(title="HNSW Vault")

# Allows the frontend (opened as a local file:// page, or served separately)
# to call this API from the browser. Wide open here since this is a local
# dev project - lock this down to specific origins before deploying anywhere.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Make sure storage directories exist before anything tries to use them
os.makedirs(config.ORIGINALS_DIR, exist_ok=True)
os.makedirs(config.THUMBNAILS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(config.INDEX_PATH), exist_ok=True)

# Load an existing index off disk if we have one, otherwise start fresh
if os.path.exists(config.INDEX_PATH):
    hnsw_index = HNSWIndex.load(config.INDEX_PATH)
else:
    hnsw_index = HNSWIndex(
        dim=config.EMBEDDING_DIM,
        M=config.HNSW_M,
        ef_construction=config.HNSW_EF_CONSTRUCTION,
        ef_search=config.HNSW_EF_SEARCH,
    )

session = db.init_db(config.DB_PATH)


@app.post("/upload")
async def upload_images(files: list[UploadFile] = File(...)):
    """Bulk upload: accepts multiple image files, runs each through the ingestion pipeline."""
    images_processed = 0
    total_faces = 0

    for file in files:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        try:
            faces_found = ingest_image(session, hnsw_index, tmp_path, config.ORIGINALS_DIR,
                                        display_filename=file.filename)
            total_faces += faces_found
            images_processed += 1
        finally:
            os.remove(tmp_path)

    hnsw_index.save(config.INDEX_PATH)  # persist so a restart doesn't lose the index

    return {
        "images_processed": images_processed,
        "faces_indexed": total_faces,
        "index_size": hnsw_index.size(),
    }


@app.post("/search")
async def search_face(file: UploadFile = File(...), top_k: int = 10):
    """Upload a single face photo, get back the images that contain a matching face."""
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        faces = get_face_embeddings(tmp_path)
        if not faces:
            return {"error": "No face detected in the query image."}

        query_embedding, _ = faces[0]  # if multiple faces in the query, use the first
        results = hnsw_index.search(query_embedding, k=top_k)

        matches = []
        for distance, hnsw_id in results:
            if distance > config.MATCH_THRESHOLD:
                continue  # too far away to be considered the same person

            image = db.get_image_by_hnsw_id(session, hnsw_id)
            if image is None or not os.path.exists(image.original_path):
                continue  # skip stale/missing files rather than failing the whole search

            thumb_path = get_or_create_thumbnail(image.original_path, config.THUMBNAILS_DIR)
            matches.append({
                "filename": image.filename,
                "distance": round(distance, 4),
                "thumbnail_url": f"/thumbnails/{os.path.basename(thumb_path)}",
                "original_url": f"/originals/{image.filename}",
            })

        return {"matches": matches}
    finally:
        os.remove(tmp_path)


@app.get("/stats")
async def stats():
    """Quick sanity-check endpoint: how much has been indexed so far."""
    return {
        "images": db.count_images(session),
        "faces": db.count_faces(session),
        "index_size": hnsw_index.size(),
    }


# Serve stored images directly so the frontend can display them
app.mount("/originals", StaticFiles(directory=config.ORIGINALS_DIR), name="originals")
app.mount("/thumbnails", StaticFiles(directory=config.THUMBNAILS_DIR), name="thumbnails")