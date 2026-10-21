"""Train or run YOLO helmet detection with the project's labeled dataset."""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "data" / "dataset" / "HelmetViolationsV2"
DATASET_YAML = PROJECT_DIR / "data" / "dataset.yaml"
DEFAULT_WEIGHTS = PROJECT_DIR / "data" / "dataset" / "best.pt"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "results" / "helmet_detection.jpg"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NO_HELMET_NAMES = {"nohelmet", "withouthelmet"}


def is_no_helmet(label):
    """Recognize common no-helmet class label spellings."""
    normalized = label.lower().replace("_", "").replace("-", "").replace(" ", "")
    return normalized in NO_HELMET_NAMES


def check_training_data():
    """Check that train and validation images have matching YOLO label files."""
    if not DATASET_YAML.is_file():
        raise FileNotFoundError(f"Dataset config not found: {DATASET_YAML}")

    for split, folder in (("train", "train"), ("val", "valid")):
        image_dir = DATASET_DIR / folder / "images"
        label_dir = DATASET_DIR / folder / "labels"
        images = [
            path for path in image_dir.glob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]

        if not images:
            raise ValueError(f"No images found in {image_dir}")

        missing_labels = [
            path.name for path in images if not (label_dir / f"{path.stem}.txt").is_file()
        ]
        if missing_labels:
            examples = ", ".join(missing_labels[:3])
            raise ValueError(
                f"{len(missing_labels)} image(s) in {split} are missing YOLO label "
                f"files in {label_dir}; examples: {examples}"
            )


def train_model(epochs, image_size):
    """Train the small YOLO model using the project's dataset configuration."""
    check_training_data()
    starting_weights = DEFAULT_WEIGHTS if DEFAULT_WEIGHTS.is_file() else "yolov8n.pt"
    model = YOLO(str(starting_weights))
    model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=image_size,
        project=str(PROJECT_DIR / "models"),
        name="helmet_detection",
        exist_ok=True,
    )
    print(
        "Training finished. Best weights: "
        f"{PROJECT_DIR / 'models' / 'helmet_detection' / 'weights' / 'best.pt'}"
    )


def detect_helmet(image_path, weights_path, output_path, confidence):
    """Detect helmet-related objects in one image and save an annotated copy."""
    image_path = Path(image_path).resolve()
    weights_path = Path(weights_path).resolve()
    output_path = Path(output_path).resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Trained weights not found: {weights_path}. Train the model first."
        )

    model = YOLO(str(weights_path))
    if not any(is_no_helmet(name) for name in model.names.values()):
        raise ValueError(
            "The selected YOLO weights do not contain a no-helmet class. "
            f"Model classes: {model.names}"
        )
    results = model.predict(source=str(image_path), conf=confidence, verbose=False)
    result = results[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result.plot()):
        raise OSError(f"Could not save annotated image: {output_path}")

    no_helmet_found = False
    for box in result.boxes:
        class_id = int(box.cls.item())
        label = result.names[class_id]
        score = float(box.conf.item())
        print(f"{label}: {score:.2f}")
        if is_no_helmet(label):
            no_helmet_found = True

    if no_helmet_found:
        print("Possible no-helmet violation detected. Review the image before taking action.")
    else:
        print("No no-helmet detection was returned. Review the image and confidence threshold.")

    print(f"Annotated image saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train or run YOLO helmet detection.")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Train YOLO on your labeled dataset.")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--image-size", type=int, default=640)

    detect_parser = commands.add_parser("detect", help="Detect helmet classes in one image.")
    detect_parser.add_argument("--image", required=True, help="Path to the test image.")
    detect_parser.add_argument(
        "--weights", default=str(DEFAULT_WEIGHTS), help="Path to trained best.pt weights."
    )
    detect_parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT), help="Where to save the annotated image."
    )
    detect_parser.add_argument("--confidence", type=float, default=0.25)

    args = parser.parse_args()
    try:
        if args.command == "train":
            if args.epochs < 1 or args.image_size < 1:
                raise ValueError("Epochs and image size must be positive numbers.")
            train_model(args.epochs, args.image_size)
        else:
            if not 0 <= args.confidence <= 1:
                raise ValueError("Confidence must be between 0 and 1.")
            detect_helmet(args.image, args.weights, args.output, args.confidence)
    except (FileNotFoundError, ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
