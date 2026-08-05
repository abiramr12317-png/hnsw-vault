"""
Metadata database: the glue between the HNSW index and actual image files.

The HNSW index only knows about vectors and integer ids - it has no concept of
"images" or "files". This module maps those integer ids back to real image
paths and face bounding boxes.
"""
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

Base = declarative_base()


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    original_path = Column(String, nullable=False)

    faces = relationship("Face", back_populates="image")


class Face(Base):
    __tablename__ = "faces"

    id = Column(Integer, primary_key=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    hnsw_id = Column(Integer, nullable=False, unique=True)  # id inside the HNSW index
    top = Column(Integer)
    right = Column(Integer)
    bottom = Column(Integer)
    left = Column(Integer)

    image = relationship("Image", back_populates="faces")


def init_db(db_path="storage/metadata.db"):
    """Creates the DB file/tables if they don't exist yet, returns a session."""
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    return session_factory()


def add_image(session, filename, original_path):
    image = Image(filename=filename, original_path=original_path)
    session.add(image)
    session.commit()
    return image.id


def add_face(session, image_id, hnsw_id, bbox):
    top, right, bottom, left = bbox
    face = Face(image_id=image_id, hnsw_id=hnsw_id, top=top, right=right, bottom=bottom, left=left)
    session.add(face)
    session.commit()
    return face.id


def get_image_by_hnsw_id(session, hnsw_id):
    """Given an id returned by the HNSW index, look up the source image."""
    face = session.query(Face).filter_by(hnsw_id=hnsw_id).first()
    return face.image if face else None


def count_images(session):
    return session.query(Image).count()


def count_faces(session):
    return session.query(Face).count()
