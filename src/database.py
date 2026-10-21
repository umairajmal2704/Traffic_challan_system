import argparse
from contextlib import contextmanager
from pathlib import Path
import sqlite3


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_DIR / "data" / "challan.db"


@contextmanager
def get_connection():
    """Open and close a connection to the project database."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database():
    """Create tables and add the helmet identity/challan links."""
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS persons (
                cnic TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                phone TEXT NOT NULL,
                email TEXT NOT NULL,
                face_embedding TEXT,
                face_image_path TEXT,
                consent_confirmed INTEGER NOT NULL DEFAULT 0
                    CHECK (consent_confirmed IN (0, 1))
            );

            CREATE TABLE IF NOT EXISTS vehicles (
                plate_no TEXT PRIMARY KEY,
                owner_cnic TEXT NOT NULL,
                FOREIGN KEY (owner_cnic) REFERENCES persons(cnic)
            );

            CREATE TABLE IF NOT EXISTS challans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                violation_type TEXT NOT NULL
                    CHECK (violation_type = 'no_helmet'),
                cnic TEXT NOT NULL,
                plate_no TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                location TEXT NOT NULL,
                fine REAL NOT NULL,
                evidence_image_path TEXT,
                message TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                detection_id INTEGER UNIQUE,
                FOREIGN KEY (cnic) REFERENCES persons(cnic),
                FOREIGN KEY (detection_id) REFERENCES helmet_detections(id)
            );

            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                challan_id INTEGER NOT NULL,
                cnic TEXT NOT NULL,
                channel TEXT NOT NULL,
                destination TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL DEFAULT 'simulated',
                FOREIGN KEY (challan_id) REFERENCES challans(id),
                FOREIGN KEY (cnic) REFERENCES persons(cnic)
            );

            CREATE TABLE IF NOT EXISTS helmet_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_path TEXT NOT NULL,
                media_type TEXT NOT NULL CHECK (media_type IN ('image', 'video')),
                frame_number INTEGER,
                track_id TEXT,
                confidence REAL NOT NULL,
                evidence_image_path TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                review_status TEXT NOT NULL DEFAULT 'pending_review'
                    CHECK (review_status IN ('pending_review', 'confirmed', 'rejected'))
            );
            """
        )
        _add_column(
            connection,
            "persons",
            "face_image_path",
            "TEXT",
        )
        _add_column(
            connection,
            "persons",
            "consent_confirmed",
            "INTEGER NOT NULL DEFAULT 0 CHECK (consent_confirmed IN (0, 1))",
        )
        _add_column(
            connection,
            "helmet_detections",
            "person_cnic",
            "TEXT REFERENCES persons(cnic)",
        )
        _add_column(
            connection,
            "helmet_detections",
            "identity_verified",
            "INTEGER NOT NULL DEFAULT 0 CHECK (identity_verified IN (0, 1))",
        )
        _add_column(
            connection,
            "challans",
            "detection_id",
            "INTEGER REFERENCES helmet_detections(id)",
        )
        connection.executescript(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_challans_detection_id
                ON challans(detection_id)
                WHERE detection_id IS NOT NULL;

            CREATE TRIGGER IF NOT EXISTS verify_helmet_person_before_update
            BEFORE UPDATE OF identity_verified ON helmet_detections
            WHEN NEW.identity_verified = 1
                AND (
                    NEW.person_cnic IS NULL
                    OR NOT EXISTS (
                        SELECT 1 FROM persons
                        WHERE cnic = NEW.person_cnic AND consent_confirmed = 1
                    )
                )
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'Verified identity must link to a person who consented to data processing.'
                );
            END;

            CREATE TRIGGER IF NOT EXISTS require_verified_helmet_challan
            BEFORE INSERT ON challans
            WHEN NEW.violation_type != 'no_helmet'
                OR NEW.detection_id IS NULL
                OR NOT EXISTS (
                    SELECT 1
                    FROM helmet_detections AS detection
                    JOIN persons AS person ON person.cnic = detection.person_cnic
                    WHERE detection.id = NEW.detection_id
                        AND detection.review_status = 'confirmed'
                        AND detection.identity_verified = 1
                        AND detection.person_cnic = NEW.cnic
                        AND person.consent_confirmed = 1
                )
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'A challan requires a human-confirmed detection linked to a consented, verified person.'
                );
            END;

            CREATE TRIGGER IF NOT EXISTS require_verified_helmet_challan_update
            BEFORE UPDATE OF violation_type, cnic, detection_id ON challans
            WHEN NEW.violation_type != 'no_helmet'
                OR NEW.detection_id IS NULL
                OR NOT EXISTS (
                    SELECT 1
                    FROM helmet_detections AS detection
                    JOIN persons AS person ON person.cnic = detection.person_cnic
                    WHERE detection.id = NEW.detection_id
                        AND detection.review_status = 'confirmed'
                        AND detection.identity_verified = 1
                        AND detection.person_cnic = NEW.cnic
                        AND person.consent_confirmed = 1
                )
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'A challan requires a human-confirmed detection linked to a consented, verified person.'
                );
            END;
            """
        )


def _add_column(connection, table, column, definition):
    """Add a column to an existing SQLite table only when it is missing."""
    columns = {
        row[1] for row in connection.execute(f"PRAGMA table_info({table})")
    }
    if column not in columns:
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def add_person(
    cnic,
    name,
    phone,
    email,
    face_image_path=None,
    consent_confirmed=False,
):
    """Save one person's details after their explicit consent was confirmed."""
    if not consent_confirmed:
        raise ValueError("The person must explicitly consent before their details are saved.")

    fields = {
        "CNIC": cnic.strip(),
        "Name": name.strip(),
        "Phone": phone.strip(),
    }
    missing = [label for label, value in fields.items() if not value]
    if missing:
        raise ValueError(f"These fields are required: {', '.join(missing)}.")
    email = email.strip()
    if email and ("@" not in email or email.startswith("@") or email.endswith("@")):
        raise ValueError("Enter a valid email address.")
    phone_digits = "".join(character for character in fields["Phone"] if character.isdigit())
    if len(phone_digits) < 7 or len(phone_digits) > 15:
        raise ValueError("Phone number must contain 7 to 15 digits.")

    resolved_face_path = None
    if face_image_path:
        image_path = Path(face_image_path.strip().strip('"').strip("'")).expanduser()
        if not image_path.is_absolute():
            image_path = PROJECT_DIR / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Face photo not found: {image_path}")
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                image.verify()
        except (ImportError, OSError) as error:
            raise ValueError(f"Face photo is not a readable image: {image_path}") from error
        resolved_face_path = str(image_path)

    initialize_database()
    try:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO persons (
                    cnic, name, phone, email, face_image_path, consent_confirmed
                )
                VALUES (?, ?, ?, ?, ?, 1)
                """,
                (
                    fields["CNIC"],
                    fields["Name"],
                    fields["Phone"],
                    email,
                    resolved_face_path,
                ),
            )
    except sqlite3.IntegrityError as error:
        if "persons.cnic" in str(error):
            raise ValueError("That CNIC is already registered.") from error
        raise

    return fields["Name"]


def add_person_interactively():
    """Ask for consent and details in the terminal; do not echo data in a command."""
    answer = input(
        "Has this person explicitly agreed to store and use their personal details "
        "and face photo for this project demo? Type yes to continue: "
    ).strip().lower()
    if answer != "yes":
        print("No person was added. Consent is required.")
        return

    cnic = input("CNIC: ").strip()
    name = input("Full name: ").strip()
    phone = input("Phone: ").strip()
    email = input("Email (optional; press Enter to skip): ").strip()
    face_image_path = input(
        "Face photo path in the project (optional; press Enter to skip): "
    ).strip()
    saved_name = add_person(
        cnic,
        name,
        phone,
        email,
        face_image_path or None,
        consent_confirmed=True,
    )
    print(f"Saved consented participant record for {saved_name}.")
    print("No face match or challan was created.")


def list_persons():
    """List enrolled names with masked CNICs, without exposing contact details."""
    initialize_database()
    with get_connection() as connection:
        people = connection.execute(
            "SELECT cnic, name, consent_confirmed, face_image_path "
            "FROM persons ORDER BY name"
        ).fetchall()

    if not people:
        print("No people are registered.")
        return
    for cnic, name, consent_confirmed, face_image_path in people:
        masked_cnic = f"****-***{cnic[-4:]}" if len(cnic) >= 4 else "****"
        consent = "consented" if consent_confirmed else "demo/unconfirmed"
        photo = "photo added" if face_image_path else "no photo"
        print(f"{name} | {masked_cnic} | {consent} | {photo}")


def insert_sample_data():
    """Insert clearly fictional demo records; repeated runs are safe."""
    sample_persons = [
        ("00000-0000000-0", "Demo Rider One", "03000000000", "rider1@example.com"),
        ("11111-1111111-1", "Demo Driver Two", "03111111111", "driver2@example.com"),
    ]
    sample_vehicles = [
        ("DEMO-001", "00000-0000000-0"),
        ("DEMO-002", "11111-1111111-1"),
    ]

    with get_connection() as connection:
        connection.executemany(
            """
            INSERT OR IGNORE INTO persons (cnic, name, phone, email)
            VALUES (?, ?, ?, ?)
            """,
            sample_persons,
        )
        connection.executemany(
            """
            INSERT OR IGNORE INTO vehicles (plate_no, owner_cnic)
            VALUES (?, ?)
            """,
            sample_vehicles,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Set up the helmet project database and manage participant records."
    )
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("init", help="Create or safely update database tables.")
    commands.add_parser("add-person", help="Interactively add a consented participant.")
    commands.add_parser("list-persons", help="List names with masked CNICs.")
    args = parser.parse_args()

    try:
        if args.command == "add-person":
            add_person_interactively()
        elif args.command == "list-persons":
            list_persons()
        else:
            initialize_database()
            print(f"Database ready: {DATABASE_PATH}")
    except (FileNotFoundError, ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
