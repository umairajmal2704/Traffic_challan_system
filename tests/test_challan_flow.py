import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import database
from challan_service import (
    FINE_AMOUNT_PKR,
    clear_unissued_detections,
    create_challan,
    get_challan_candidates,
)


class ChallanFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.original_database_path = database.DATABASE_PATH
        self.addCleanup(self.restore_database_path)
        database.DATABASE_PATH = Path(self.temp_dir.name) / "test_challan.db"
        database.initialize_database()

        with database.get_connection() as connection:
            connection.executemany(
                """
                INSERT INTO persons (
                    cnic, name, phone, email, consent_confirmed
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    ("12345-1234567-1", "Consented Rider", "03001234567", "", 1),
                    ("12345-7654321-1", "Unconsented Rider", "03007654321", "", 0),
                ],
            )

    def restore_database_path(self):
        database.DATABASE_PATH = self.original_database_path

    def add_detection(self, review_status):
        with database.get_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO helmet_detections (
                    source_path, media_type, confidence,
                    evidence_image_path, review_status
                )
                VALUES ('sample.jpg', 'image', 0.95, 'evidence.jpg', ?)
                """,
                (review_status,),
            )
            return cursor.lastrowid

    def create_challan(self, detection_id, cnic="12345-1234567-1"):
        return create_challan(
            detection_id=detection_id,
            cnic=cnic,
            location="Demo Road",
            payment_instructions="Use the demo payment desk.",
            appeal_instructions="Contact the demo office with the evidence.",
        )

    def test_confirmed_detection_saves_person_challan_and_simulated_sms(self):
        detection_id = self.add_detection("confirmed")

        result = self.create_challan(detection_id)

        self.assertEqual(result["name"], "Consented Rider")
        self.assertEqual(result["phone"], "03001234567")
        self.assertIn("Consented Rider", result["message"])
        self.assertIn("DEMO TRAFFIC CHALLAN", result["message"])
        with database.get_connection() as connection:
            challan = connection.execute(
                """
                SELECT violation_type, cnic, plate_no, status, detection_id, message
                FROM challans
                """
            ).fetchone()
            notification = connection.execute(
                "SELECT channel, destination, status FROM notifications"
            ).fetchone()
            detection = connection.execute(
                """
                SELECT person_cnic, identity_verified
                FROM helmet_detections WHERE id = ?
                """,
                (detection_id,),
            ).fetchone()

        self.assertEqual(
            challan[:5],
            ("no_helmet", "12345-1234567-1", "", "pending", detection_id),
        )
        self.assertNotIn("Vehicle plate", challan[5])
        self.assertEqual(
            notification,
            ("simulated_sms", "03001234567", "simulated"),
        )
        self.assertEqual(detection, ("12345-1234567-1", 1))

    def test_pending_detection_cannot_create_challan(self):
        detection_id = self.add_detection("pending_review")

        with self.assertRaisesRegex(ValueError, "human-confirmed"):
            self.create_challan(detection_id)

        self.assertEqual(self.count_rows("challans"), 0)

    def test_rejected_detection_cannot_create_challan(self):
        detection_id = self.add_detection("rejected")

        with self.assertRaisesRegex(ValueError, "human-confirmed"):
            self.create_challan(detection_id)

        self.assertEqual(self.count_rows("challans"), 0)

    def test_unconsented_person_cannot_receive_challan(self):
        detection_id = self.add_detection("confirmed")

        with self.assertRaisesRegex(ValueError, "has not consented"):
            self.create_challan(detection_id, "12345-7654321-1")

        self.assertEqual(self.count_rows("challans"), 0)

    def test_same_detection_cannot_receive_two_challans(self):
        detection_id = self.add_detection("confirmed")
        self.create_challan(detection_id)

        with self.assertRaisesRegex(ValueError, "already exists"):
            self.create_challan(detection_id)

        self.assertEqual(self.count_rows("challans"), 1)
        self.assertEqual(self.count_rows("notifications"), 1)

    def test_challan_uses_fixed_fine_without_user_input(self):
        detection_id = self.add_detection("confirmed")

        self.create_challan(detection_id)

        with database.get_connection() as connection:
            fine = connection.execute("SELECT fine FROM challans").fetchone()[0]
        self.assertEqual(fine, FINE_AMOUNT_PKR)
        self.assertIn("PKR 1500.00", self.get_message())

    def test_candidates_include_only_confirmed_unissued_detections(self):
        self.add_detection("pending_review")
        self.add_detection("rejected")
        confirmed_id = self.add_detection("confirmed")

        candidates = get_challan_candidates()
        self.assertEqual([row["id"] for row in candidates], [confirmed_id])

        self.create_challan(confirmed_id)
        self.assertEqual(get_challan_candidates(), [])

    def test_clear_candidates_removes_unissued_and_keeps_challan_evidence(self):
        self.add_detection("pending_review")
        self.add_detection("rejected")
        linked_detection_id = self.add_detection("confirmed")
        self.create_challan(linked_detection_id)

        deleted_count = clear_unissued_detections()

        self.assertEqual(deleted_count, 2)
        with database.get_connection() as connection:
            remaining = connection.execute(
                "SELECT id FROM helmet_detections"
            ).fetchall()
            challan = connection.execute(
                "SELECT detection_id, evidence_image_path FROM challans"
            ).fetchone()
        self.assertEqual([row[0] for row in remaining], [linked_detection_id])
        self.assertEqual(challan, (linked_detection_id, "evidence.jpg"))
        self.assertEqual(self.count_rows("challans"), 1)
        self.assertEqual(self.count_rows("notifications"), 1)

    @staticmethod
    def count_rows(table_name):
        with database.get_connection() as connection:
            return connection.execute(
                f"SELECT COUNT(*) FROM {table_name}"
            ).fetchone()[0]

    @staticmethod
    def get_message():
        with database.get_connection() as connection:
            return connection.execute("SELECT message FROM challans").fetchone()[0]


if __name__ == "__main__":
    unittest.main(verbosity=2)
