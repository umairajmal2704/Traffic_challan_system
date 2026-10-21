"""Create challans only for reviewed detections with a confirmed person match."""

from datetime import datetime
from decimal import Decimal
import sqlite3

from challan_message import generate_messages
from database import get_connection, initialize_database

FINE_AMOUNT_PKR = 1500


def get_challan_candidates():
    """Return confirmed detections that do not already have a challan."""
    initialize_database()
    with get_connection() as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(
            """
            SELECT detection.id, detection.source_path, detection.frame_number,
                   detection.track_id, detection.confidence,
                   detection.evidence_image_path, detection.created_at
            FROM helmet_detections AS detection
            LEFT JOIN challans ON challans.detection_id = detection.id
            WHERE detection.review_status = 'confirmed'
                AND challans.id IS NULL
            ORDER BY detection.id DESC
            """
        ).fetchall()


def clear_unissued_detections():
    """Delete detection history that is not referenced by an issued challan."""
    initialize_database()
    with get_connection() as connection:
        cursor = connection.execute(
            """
            DELETE FROM helmet_detections
            WHERE NOT EXISTS (
                SELECT 1 FROM challans
                WHERE challans.detection_id = helmet_detections.id
            )
            """
        )
        return cursor.rowcount


def get_consented_person(cnic):
    """Fetch the contact details for a person who consented to the demo."""
    with get_connection() as connection:
        connection.row_factory = sqlite3.Row
        person = connection.execute(
            """
            SELECT cnic, name, phone
            FROM persons
            WHERE cnic = ? AND consent_confirmed = 1
            """,
            (cnic,),
        ).fetchone()
    if person is None:
        raise ValueError("The matched person is missing or has not consented.")
    return person


def create_challan(
    detection_id,
    cnic,
    location,
    payment_instructions,
    appeal_instructions,
):
    """Save a bilingual challan and a simulated phone notification."""
    location = location.strip()
    payment_instructions = payment_instructions.strip()
    appeal_instructions = appeal_instructions.strip()
    if not location or not payment_instructions or not appeal_instructions:
        raise ValueError(
            "Location, payment instructions, and appeal instructions are required."
        )

    fine_amount = Decimal(FINE_AMOUNT_PKR)

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with get_connection() as connection:
        connection.row_factory = sqlite3.Row
        detection = connection.execute(
            """
            SELECT review_status, evidence_image_path
            FROM helmet_detections
            WHERE id = ?
            """,
            (detection_id,),
        ).fetchone()
        if detection is None:
            raise ValueError("The selected detection no longer exists.")
        if detection["review_status"] != "confirmed":
            raise ValueError("Only a human-confirmed detection can receive a challan.")

        person = connection.execute(
            """
            SELECT name, phone
            FROM persons
            WHERE cnic = ? AND consent_confirmed = 1
            """,
            (cnic,),
        ).fetchone()
        if person is None:
            raise ValueError("The matched person is missing or has not consented.")

        message_parts = generate_messages(
            violation_type="no_helmet",
            timestamp=timestamp,
            location=location,
            fine=fine_amount,
            payment=payment_instructions,
            appeal=appeal_instructions,
            person_name=person["name"],
        )
        message = (
            f"--- ENGLISH ---\n{message_parts['english']}\n\n"
            f"--- URDU ---\n{message_parts['urdu']}"
        )

        cursor = connection.execute(
            """
            UPDATE helmet_detections
            SET person_cnic = ?, identity_verified = 1
            WHERE id = ? AND review_status = 'confirmed'
            """,
            (cnic, detection_id),
        )
        if cursor.rowcount != 1:
            raise ValueError("The detection could not be linked to this person.")

        try:
            challan_cursor = connection.execute(
                """
                INSERT INTO challans (
                    violation_type, cnic, plate_no, timestamp, location, fine,
                    evidence_image_path, message, status, detection_id
                )
                VALUES ('no_helmet', ?, '', ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    cnic,
                    timestamp,
                    location,
                    float(fine_amount),
                    detection["evidence_image_path"],
                    message,
                    detection_id,
                ),
            )
        except sqlite3.IntegrityError as error:
            if "UNIQUE constraint failed" in str(error):
                raise ValueError("A challan already exists for this detection.") from error
            raise

        challan_id = challan_cursor.lastrowid
        connection.execute(
            """
            INSERT INTO notifications (challan_id, cnic, channel, destination, message)
            VALUES (?, ?, 'simulated_sms', ?, ?)
            """,
            (challan_id, cnic, person["phone"], message),
        )

    return {
        "challan_id": challan_id,
        "name": person["name"],
        "phone": person["phone"],
        "message": message,
    }
