"""Run helmet detection on an image or video and save reviewable candidates."""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from contextlib import nullcontext
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from database import get_connection, initialize_database
from detect_helmet import is_no_helmet


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_DIR / "data" / "dataset" / "best.pt"
DEFAULT_RESULTS = PROJECT_DIR / "data" / "results" / "helmet_review"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
@dataclass
class Candidate:
    track_id: str
    frame_number: int
    confidence: float
    evidence_frame: object


def save_image_candidates(model, source_path, output_dir, confidence, no_helmet_ids):
    result = model.predict(
        source=str(source_path),
        conf=confidence,
        classes=no_helmet_ids,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
    )[0]
    annotated = result.plot()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_image = output_dir / f"{source_path.stem}_annotated{source_path.suffix}"
    if not cv2.imwrite(str(output_image), annotated):
        raise OSError(f"Could not save annotated image: {output_image}")

    candidates = []
    for index, box in enumerate(result.boxes):
        label = result.names[int(box.cls.item())]
        if is_no_helmet(label):
            candidates.append(
                Candidate(
                    track_id=f"image-{index + 1}",
                    frame_number=0,
                    confidence=float(box.conf.item()),
                    evidence_frame=annotated,
                )
            )
    return candidates, output_image


def save_video_candidates(
    model,
    source_path,
    output_dir,
    confidence,
    no_helmet_ids,
    frame_stride=1,
):
    if frame_stride < 1:
        raise ValueError("Frame stride must be at least 1.")

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {source_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if width < 1 or height < 1:
        capture.release()
        raise ValueError(f"Video has invalid dimensions: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / f"{source_path.stem}_annotated.mp4"
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        (fps if fps > 0 else 25.0) / frame_stride,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not create annotated video: {output_video}")

    candidates = {}
    frame_number = 0
    device = 0 if torch.cuda.is_available() else "cpu"
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number % frame_stride:
                frame_number += 1
                continue

            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if torch.cuda.is_available()
                else nullcontext()
            )
            with autocast:
                result = model.track(
                    source=frame,
                    persist=True,
                    conf=confidence,
                    classes=no_helmet_ids,
                    device=device,
                    verbose=False,
                )[0]
            annotated = result.plot()
            writer.write(annotated)

            boxes = result.boxes
            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
            else:
                track_ids = [None] * len(boxes)

            for index, box in enumerate(boxes):
                label = result.names[int(box.cls.item())]
                if not is_no_helmet(label):
                    continue
                track_id = track_ids[index]
                candidate_key = (
                    f"track-{track_id}" if track_id is not None
                    else f"frame-{frame_number}-detection-{index}"
                )
                score = float(box.conf.item())
                current = candidates.get(candidate_key)
                if current is None or score > current.confidence:
                    candidates[candidate_key] = Candidate(
                        track_id=str(track_id) if track_id is not None else "",
                        frame_number=frame_number,
                        confidence=score,
                        evidence_frame=annotated.copy(),
                    )
            frame_number += 1
            if frame_number % 300 == 0:
                print(f"Processed {frame_number} frame(s).")
    finally:
        capture.release()
        writer.release()

    if frame_number == 0:
        output_video.unlink(missing_ok=True)
        raise ValueError(f"No readable frames found in video: {source_path}")
    return list(candidates.values()), output_video


def save_candidates(candidates, source_path, media_type, evidence_dir, report_path):
    evidence_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    records = []
    for index, candidate in enumerate(candidates, start=1):
        evidence_path = evidence_dir / (
            f"{source_path.stem}_{timestamp}_candidate_{index:03d}.jpg"
        )
        if not cv2.imwrite(str(evidence_path), candidate.evidence_frame):
            raise OSError(f"Could not save candidate evidence: {evidence_path}")
        records.append(
            {
                "source_path": str(source_path),
                "media_type": media_type,
                "frame_number": candidate.frame_number,
                "track_id": candidate.track_id,
                "confidence": candidate.confidence,
                "evidence_image_path": str(evidence_path),
                "review_status": "pending_review",
            }
        )

    with get_connection() as connection:
        connection.executemany(
            """
            INSERT INTO helmet_detections (
                source_path, media_type, frame_number, track_id,
                confidence, evidence_image_path, review_status
            )
            VALUES (
                :source_path, :media_type, :frame_number, :track_id,
                :confidence, :evidence_image_path, :review_status
            )
            """,
            records,
        )

    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_path",
                "media_type",
                "frame_number",
                "track_id",
                "confidence",
                "evidence_image_path",
                "review_status",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    return records


def process_media(media_path, weights_path, output_dir, confidence, frame_stride=1):
    media_path = Path(media_path).resolve()
    weights_path = Path(weights_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not media_path.is_file():
        raise FileNotFoundError(f"Input media not found: {media_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    suffix = media_path.suffix.lower()
    model = YOLO(str(weights_path))
    if not any(is_no_helmet(name) for name in model.names.values()):
        raise ValueError(
            "The selected YOLO weights do not contain a no-helmet class. "
            f"Model classes: {model.names}"
        )
    no_helmet_ids = [
        class_id for class_id, name in model.names.items() if is_no_helmet(name)
    ]
    if suffix in IMAGE_SUFFIXES:
        candidates, annotated_path = save_image_candidates(
            model, media_path, output_dir, confidence, no_helmet_ids
        )
        media_type = "image"
    elif suffix in VIDEO_SUFFIXES:
        candidates, annotated_path = save_video_candidates(
            model,
            media_path,
            output_dir,
            confidence,
            no_helmet_ids,
            frame_stride=frame_stride,
        )
        media_type = "video"
    else:
        raise ValueError(f"Unsupported media format: {suffix or '(no extension)'}")

    evidence_dir = output_dir / "evidence"
    report_path = output_dir / f"{media_path.stem}_candidates.csv"
    records = save_candidates(
        candidates, media_path, media_type, evidence_dir, report_path
    )

    print(f"Annotated media saved to: {annotated_path}")
    print(f"Possible no-helmet candidates: {len(records)}")
    print(f"Review report saved to: {report_path}")
    if records:
        print(f"Candidate evidence saved to: {evidence_dir}")
    print("Candidates are pending human review; no identity or challan was assigned.")


def main():
    parser = argparse.ArgumentParser(
        description="Detect possible no-helmet cases and save them for human review."
    )
    parser.add_argument("--input", required=True, help="Path to an image or video.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Run detection on every Nth video frame. Use 1 for all frames.",
    )
    args = parser.parse_args()

    if not 0 <= args.confidence <= 1:
        parser.error("Confidence must be between 0 and 1.")
    if args.frame_stride < 1:
        parser.error("Frame stride must be at least 1.")
    try:
        initialize_database()
        process_media(
            args.input,
            args.weights,
            args.output_dir,
            args.confidence,
            frame_stride=args.frame_stride,
        )
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
