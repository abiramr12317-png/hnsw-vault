"""Central place for tunable parameters and file paths."""

# HNSW parameters
HNSW_M = 16                    # max neighbors per node per layer (higher = better recall, more memory)
HNSW_EF_CONSTRUCTION = 200     # candidate list size while building the graph
HNSW_EF_SEARCH = 50            # candidate list size while searching (higher = better recall, slower)

# Face match threshold (euclidean distance on face_recognition's 128-d embeddings).
# face_recognition's own convention: distance < 0.55 usually means "same person".
MATCH_THRESHOLD = 0.065

# Storage paths (relative to project root)
ORIGINALS_DIR = "storage/originals"
THUMBNAILS_DIR = "storage/thumbnails"
INDEX_PATH = "storage/index_data/hnsw_index.pkl"
DB_PATH = "storage/metadata.db"

EMBEDDING_DIM = 128  # face_recognition's default embedding size
