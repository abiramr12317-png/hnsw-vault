"""
Wraps the face_recognition library (dlib-based) for face detection and
embedding extraction. This is the one part of the project that's a library
call rather than something built from scratch - and that's intentional.
"""
import face_recognition


def get_face_embeddings(image_path):
    """
    Detects every face in an image and generates its embedding.

    Returns a list of (embedding, bbox) tuples - one per detected face.
    embedding is a 128-d numpy array. bbox is (top, right, bottom, left)
    in pixel coordinates, matching face_recognition's convention.
    """
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    encodings = face_recognition.face_encodings(image, face_locations)
    return list(zip(encodings, face_locations))
