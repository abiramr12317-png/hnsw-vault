# HNSW Vault

A face-search system for large event photo collections. Upload a folder of
event photos once, then upload a single photo of a person's face at any point
after to retrieve every photo they appear in.

Face detection and embedding generation use a pretrained model
(`face_recognition` / dlib) - that part is a library call, not original work.
The actual engineering contribution is everything around it: a **from-scratch
HNSW (Hierarchical Navigable Small World) index** for fast approximate
nearest-neighbor search over the face embeddings, plus the storage and
metadata layer that connects search results back to real image files.

## Architecture

```
Ingestion:  upload -> face detection -> embeddings -> HNSW insert -> metadata DB
Query:      query face -> embedding -> HNSW search -> metadata lookup -> thumbnails
```

Thumbnails are generated lazily - only for images that actually show up in a
search result, not for every uploaded photo - and cached on disk after the
first generation.

## Project structure

```
hnsw-vault/
├── requirements.txt
├── app/
│   ├── main.py          FastAPI app: /upload, /search, /stats
│   ├── config.py        tunable parameters (HNSW params, match threshold, paths)
│   ├── hnsw.py           the custom HNSW implementation
│   ├── embeddings.py     wraps face_recognition (the one library-dependent piece)
│   ├── thumbnails.py     lazy thumbnail generation + caching
│   ├── pipeline.py       orchestrates ingestion end to end
│   └── db.py             metadata DB (SQLAlchemy models + helpers)
├── storage/
│   ├── originals/        full-resolution uploaded images
│   ├── thumbnails/       generated on demand
│   └── index_data/       persisted HNSW graph (pickled)
├── tests/
│   └── test_hnsw.py       recall, speed, and edge-case tests for the index
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
```

## Setup

```bash
pip install -r requirements.txt
```

Note: `dlib` compiles from C++ source and can take several minutes to install
(it needs CMake and a C++ compiler present on your system first). This is a
known friction point with `face_recognition` - if it fails, check that CMake
is installed, or consider `insightface` as an alternative embedding library.

## Running it

```bash
uvicorn app.main:app --reload
```

Then open `frontend/index.html` in a browser (or serve it with any static
file server) - it talks to the API at `http://localhost:8000`.

## Testing the index

The HNSW implementation is the part of this project worth actually
validating, since a broken implementation still runs without crashing - it
just quietly returns wrong results. Run:

```bash
pytest tests/test_hnsw.py -v -s
```

**Results from this implementation, tested on synthetic 32-dimensional
vectors** (same dimensionality class as real face embeddings):

- **Recall@10: 100%** across 30 random queries against a 500-vector index,
  compared against brute-force ground truth.
- **Query latency vs. brute force:**

  | dataset size | brute-force | HNSW |
  |---|---|---|
  | 200 | 0.37 ms | 0.43 ms |
  | 1,000 | 1.80 ms | 1.15 ms |
  | 3,000 | 8.96 ms | 1.70 ms |

  HNSW is roughly on par with brute force at very small scale (graph
  traversal overhead dominates), and pulls clearly ahead as the dataset
  grows - about **5x faster at 3,000 vectors**, with the gap widening as
  more images are added.

Also covered: empty-index search, single-insert self-retrieval, k larger
than the dataset, and duplicate-vector handling.

## Tuning

Key parameters live in `app/config.py`:

- `HNSW_M` - max neighbors per node (higher = better recall, more memory)
- `HNSW_EF_CONSTRUCTION` - candidate list size while building the graph
- `HNSW_EF_SEARCH` - candidate list size while searching (higher = better
  recall, slower search)
- `MATCH_THRESHOLD` - euclidean distance cutoff below which two faces are
  considered a match (0.6 is `face_recognition`'s own convention; tune this
  against your own labeled photos for best results)
