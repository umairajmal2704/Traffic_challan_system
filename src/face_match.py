import argparse
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_DIR / "data" / "challan.db"
MATCH_THRESHOLD = 0.8
MATCH_MARGIN = 0.15


@contextmanager
def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _normalize_path(path_value):
    path_value = str(path_value).strip().strip('"').strip("'")
    if not path_value:
        return None
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = (PROJECT_DIR / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return str(candidate)


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionResnetV1(pretrained="vggface2").to(device).eval()
    face_detector = MTCNN(keep_all=True, device=device)
    return model, face_detector


def extract_embedding(image_path, model=None, face_detector=None):
    if isinstance(image_path, Image.Image):
        image = image_path.convert("RGB")
        image_label = "uploaded image"
    else:
        image_path = Path(image_path).expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image_label = str(image_path)

    if model is None or face_detector is None:
        model, face_detector = load_model()

    face = face_detector(image)
    if face is None:
        raise ValueError(f"No face was detected in {image_label}.")
    if face.ndim == 4:
        if face.shape[0] != 1:
            raise ValueError(
                f"Found {face.shape[0]} faces in {image_label}. "
                "Use a clear crop containing only the rider's face."
            )
        face = face[0]

    face = face.to(next(model.parameters()).device)
    with torch.no_grad():
        embedding = model(face.unsqueeze(0))[0].detach().cpu().numpy()
    return embedding.astype(float)


def store_embedding_for_person(cnic, image_path):
    with get_connection() as connection:
        person = connection.execute(
            "SELECT consent_confirmed FROM persons WHERE cnic = ?",
            (cnic,),
        ).fetchone()
    if person is None:
        raise ValueError(f"No person is registered for CNIC {cnic}.")
    if not person["consent_confirmed"]:
        raise ValueError("Face enrollment requires the person's consent.")

    model, face_detector = load_model()
    embedding = extract_embedding(image_path, model=model, face_detector=face_detector)
    with get_connection() as connection:
        cursor = connection.execute(
            "UPDATE persons SET face_embedding = ? WHERE cnic = ?",
            (json.dumps(embedding.tolist()), cnic),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"No person was updated for CNIC {cnic}.")
    return embedding


def refresh_all_embeddings():
    with get_connection() as connection:
        people = connection.execute(
            """
            SELECT cnic, face_image_path FROM persons
            WHERE face_image_path IS NOT NULL AND consent_confirmed = 1
            """
        ).fetchall()

    for person in people:
        image_path = _normalize_path(person["face_image_path"])
        if not image_path:
            continue
        try:
            store_embedding_for_person(person["cnic"], image_path)
            print(f"Stored embedding for {person['cnic']} from {image_path}")
        except Exception as error:
            print(f"Skipped {person['cnic']}: {error}")


def load_embedding(json_text):
    if not json_text:
        return None
    values = json.loads(json_text)
    return np.asarray(values, dtype=np.float32)


def match_person_image(image_path, threshold=MATCH_THRESHOLD):
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Face match threshold must be a finite number greater than zero.")

    model, face_detector = load_model()
    query_embedding = extract_embedding(image_path, model=model, face_detector=face_detector)

    with get_connection() as connection:
        people = connection.execute(
            """
            SELECT cnic, name, face_embedding FROM persons
            WHERE face_embedding IS NOT NULL AND consent_confirmed = 1
            """
        ).fetchall()

    if not people:
        raise ValueError("No enrolled faces are available in the database yet.")

    matches = []
    for person in people:
        stored_embedding = load_embedding(person["face_embedding"])
        if stored_embedding is None:
            continue
        distance = float(np.linalg.norm(query_embedding - stored_embedding))
        matches.append((distance, person["cnic"], person["name"]))

    if not matches:
        raise ValueError("No valid face embeddings are available in the database.")

    matches.sort(key=lambda item: item[0])
    best_distance, best_cnic, best_name = matches[0]
    if best_distance > threshold:
        return None, best_distance, matches[:3]
    if len(matches) > 1 and matches[1][0] - best_distance < MATCH_MARGIN:
        return None, best_distance, matches[:3]
    return best_cnic, best_distance, matches[:3]


def main():
    parser = argparse.ArgumentParser(description="Enroll faces and match a rider image to a database person.")
    commands = parser.add_subparsers(dest="command", required=True)

    refresh_parser = commands.add_parser("refresh", help="Generate face embeddings for all registered people with photos.")

    match_parser = commands.add_parser("match", help="Match a rider face image to the database.")
    match_parser.add_argument("--image", required=True, help="Path to the rider image to match.")
    match_parser.add_argument(
        "--threshold",
        type=float,
        default=MATCH_THRESHOLD,
        help=f"Maximum face distance for a valid match (default: {MATCH_THRESHOLD}).",
    )

    enroll_parser = commands.add_parser("enroll", help="Generate an embedding for one person.")
    enroll_parser.add_argument("--cnic", required=True)
    enroll_parser.add_argument("--image", required=True)

    args = parser.parse_args()

    try:
        if args.command == "refresh":
            refresh_all_embeddings()
        elif args.command == "enroll":
            embedding = store_embedding_for_person(args.cnic, args.image)
            print(f"Embedded face saved for {args.cnic}; vector size = {embedding.shape[0]}")
        else:
            cnic, distance, candidates = match_person_image(args.image, threshold=args.threshold)
            if cnic is None:
                print(f"No valid match found. Best distance = {distance:.4f}")
                for candidate_distance, candidate_cnic, candidate_name in candidates:
                    print(f"{candidate_name} | {candidate_cnic} | dist={candidate_distance:.4f}")
            else:
                print(f"Matched person: {cnic} ({distance:.4f})")
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
