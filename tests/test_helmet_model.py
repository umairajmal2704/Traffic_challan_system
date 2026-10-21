import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from detect_helmet import DEFAULT_WEIGHTS, is_no_helmet
from ultralytics import YOLO


POSITIVE_IMAGE = (
    PROJECT_DIR
    / "data"
    / "dataset"
    / "HelmetViolationsV2"
    / "test"
    / "images"
    / "1111_jpg.rf.34849ea019d0a1a70bde873ffd4fde0f.jpg"
)
CONTROL_IMAGE = (
    PROJECT_DIR
    / "data"
    / "dataset"
    / "HelmetViolationsV2"
    / "test"
    / "images"
    / "Motorcycle_Riding_Laws_and_Requirements_in_the_US_1920x1000_jpg.rf.c06df378f5bcd6987a3665f7f6531fbf.jpg"
)


@unittest.skipUnless(
    DEFAULT_WEIGHTS.is_file() and POSITIVE_IMAGE.is_file() and CONTROL_IMAGE.is_file(),
    "Project YOLO weights or held-out test images are missing.",
)
class HelmetModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = YOLO(str(DEFAULT_WEIGHTS))

    def get_predicted_labels(self, image_path):
        result = self.model.predict(
            source=str(image_path),
            conf=0.25,
            verbose=False,
        )[0]
        return [
            result.names[int(box.cls.item())]
            for box in result.boxes
        ]

    def test_model_has_a_no_helmet_class(self):
        self.assertTrue(any(is_no_helmet(name) for name in self.model.names.values()))

    def test_labeled_no_helmet_image_is_detected(self):
        labels = self.get_predicted_labels(POSITIVE_IMAGE)
        self.assertTrue(
            any(is_no_helmet(label) for label in labels),
            f"No no-helmet prediction found; predicted labels were {labels}.",
        )

    def test_helmet_wearing_control_has_no_no_helmet_prediction(self):
        labels = self.get_predicted_labels(CONTROL_IMAGE)
        self.assertFalse(
            any(is_no_helmet(label) for label in labels),
            f"False no-helmet prediction on helmet control; labels were {labels}.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
