"""Streamlit dashboard for running and reviewing helmet detections."""

import shutil
import sqlite3
import sys
from io import BytesIO
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from database import DATABASE_PATH, get_connection, initialize_database
from challan_service import (
    FINE_AMOUNT_PKR,
    clear_unissued_detections,
    create_challan,
    get_challan_candidates,
    get_consented_person,
)
from face_match import MATCH_THRESHOLD, match_person_image
from main_pipeline import DEFAULT_WEIGHTS, IMAGE_SUFFIXES, VIDEO_SUFFIXES, process_media


RESULTS_DIR = PROJECT_DIR / "data" / "results" / "dashboard_runs"
UPLOADS_DIR = PROJECT_DIR / "data" / "uploads"
PAGE_SIZE = 10

st.set_page_config(page_title="Helmet Review Dashboard", page_icon="🪖", layout="wide")


def get_review_counts():
    with get_connection() as connection:
        return dict(
            connection.execute(
                """
                SELECT review_status, COUNT(*)
                FROM helmet_detections
                GROUP BY review_status
                """
            ).fetchall()
        )


def get_candidates(status, offset, limit):
    with get_connection() as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(
            """
            SELECT id, source_path, media_type, frame_number, track_id,
                   confidence, evidence_image_path, created_at, review_status
            FROM helmet_detections
            WHERE review_status = ?
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (status, limit, offset),
        ).fetchall()


def update_review_status(candidate_id, status):
    if status not in {"confirmed", "rejected"}:
        raise ValueError("Review status must be confirmed or rejected.")
    with get_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE helmet_detections
            SET review_status = ?
            WHERE id = ? AND review_status = 'pending_review'
            """,
            (status, candidate_id),
        )
        if cursor.rowcount != 1:
            raise ValueError("This candidate was already reviewed. Refresh the page.")


def save_upload(uploaded_file):
    safe_name = Path(uploaded_file.name).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
        raise ValueError("Choose a supported image or video file.")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = UPLOADS_DIR / f"{timestamp}_{safe_name}"
    uploaded_file.seek(0)
    with destination.open("wb") as output_file:
        shutil.copyfileobj(uploaded_file, output_file, length=1024 * 1024)
    return destination


def run_detection(media_path, frame_stride=5):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = RESULTS_DIR / f"{media_path.stem}_{timestamp}"
    process_media(
        media_path,
        DEFAULT_WEIGHTS,
        output_dir,
        confidence=0.35,
        frame_stride=frame_stride,
    )
    if media_path.suffix.lower() in VIDEO_SUFFIXES:
        return output_dir / f"{media_path.stem}_annotated.mp4"
    return output_dir / f"{media_path.stem}_annotated{media_path.suffix.lower()}"


def show_detection_form():
    st.subheader("Run helmet detection")
    source_mode = st.radio(
        "Choose input",
        ("Use a video in the project", "Upload an image or video"),
        horizontal=True,
    )

    selected_path = None
    uploaded_file = None
    if source_mode == "Use a video in the project":
        video_dir = PROJECT_DIR / "data" / "videos"
        video_paths = sorted(
            path for path in video_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ) if video_dir.is_dir() else []
        if not video_paths:
            st.info("No supported videos found in data\\videos.")
        else:
            labels = {
                str(path): (
                    f"{path.name} (merged demo)" if path.name == "video5.mp4"
                    else path.name
                )
                for path in video_paths
            }
            selected_path = st.selectbox(
                "Project video",
                options=[str(path) for path in video_paths],
                format_func=lambda value: labels[value],
            )
            st.caption(f"Selected video: {selected_path}")
    else:
        uploaded_file = st.file_uploader(
            "Choose an image or video",
            type=sorted({suffix.lstrip(".") for suffix in IMAGE_SUFFIXES | VIDEO_SUFFIXES}),
        )
        if uploaded_file is not None:
            st.caption(
                f"Selected file: {uploaded_file.name} "
                f"({uploaded_file.size / (1024 * 1024):.1f} MB)"
            )

    with st.form("detection_form"):
        frame_stride = st.selectbox(
            "Video speed",
            options=(1, 3, 5, 10),
            index=2,
            format_func=lambda value: (
                "Every frame (slowest, most thorough)"
                if value == 1
                else f"Every {value} frames (faster; may miss brief events)"
            ),
            help=(
                "For a complete review, choose every frame. Sampling fewer frames "
                "reduces processing time but can miss brief no-helmet appearances."
            ),
        )
        submitted = st.form_submit_button(
            "Run detection",
            type="primary",
            disabled=selected_path is None and uploaded_file is None,
        )

    if submitted:
        try:
            if uploaded_file is not None:
                media_path = save_upload(uploaded_file)
            else:
                media_path = Path(selected_path)
            with st.spinner(
                (
                    f"Detecting on every {frame_stride} frame(s). "
                    "GPU acceleration is enabled when available."
                )
            ):
                output_path = run_detection(media_path, frame_stride=frame_stride)
            st.session_state["latest_output"] = str(output_path)
            st.session_state["latest_input"] = str(media_path)
            st.session_state["latest_stride"] = frame_stride
            st.success("Detection finished. All candidates need human review.")
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
            st.error(f"Detection could not be completed: {error}")

    output_path = st.session_state.get("latest_output")
    if output_path and Path(output_path).is_file():
        media_path = Path(st.session_state["latest_input"])
        st.markdown("#### Latest annotated result")
        if media_path.suffix.lower() in VIDEO_SUFFIXES:
            st.video(output_path)
        else:
            st.image(output_path, use_container_width=True)
        st.caption(f"Saved at: {output_path}")
        if media_path.suffix.lower() in VIDEO_SUFFIXES:
            stride = st.session_state.get("latest_stride", 1)
            if stride > 1:
                st.caption(
                    f"Detection was run on every {stride}th frame. "
                    "This is faster but may miss brief events; use every frame "
                    "when you need the most thorough review."
                )


def show_candidate_review():
    st.subheader("Review detection candidates")
    with st.expander("Reset candidate history for a fresh demo"):
        st.warning(
            "This permanently removes all detections without a challan, including "
            "pending, confirmed, and rejected candidates. Existing challans and "
            "their linked detections are kept."
        )
        confirm_reset = st.checkbox(
            "I understand this cannot be undone.",
            key="confirm_clear_candidates",
        )
        if st.button(
            "Clear candidates without challans",
            type="secondary",
            disabled=not confirm_reset,
            key="clear_unissued_candidates",
        ):
            try:
                deleted_count = clear_unissued_detections()
                st.session_state.pop("confirm_clear_candidates", None)
                st.session_state["candidate_reset_message"] = (
                    f"Cleared {deleted_count} candidate(s) without a challan. "
                    "Existing challans were preserved."
                )
                st.rerun()
            except (sqlite3.Error, OSError, ValueError) as error:
                st.error(f"Could not clear candidate history: {error}")

    reset_message = st.session_state.pop("candidate_reset_message", None)
    if reset_message:
        st.success(reset_message)

    counts = get_review_counts()
    pending_count = counts.get("pending_review", 0)
    confirmed_count = counts.get("confirmed", 0)
    rejected_count = counts.get("rejected", 0)

    metric_columns = st.columns(3)
    metric_columns[0].metric("Pending review", pending_count)
    metric_columns[1].metric("Confirmed by reviewer", confirmed_count)
    metric_columns[2].metric("Rejected by reviewer", rejected_count)

    status = st.selectbox(
        "Show candidates",
        ("pending_review", "confirmed", "rejected"),
        format_func=lambda value: {
            "pending_review": "Pending review",
            "confirmed": "Confirmed by reviewer",
            "rejected": "Rejected by reviewer",
        }[value],
    )
    total = counts.get(status, 0)
    if not total:
        st.info("No candidates in this review category.")
        return

    page_count = (total + PAGE_SIZE - 1) // PAGE_SIZE
    page = st.number_input(
        f"Page (1–{page_count})",
        min_value=1,
        max_value=page_count,
        value=1,
        step=1,
        key=f"review_page_{status}",
    )
    candidates = get_candidates(status, (page - 1) * PAGE_SIZE, PAGE_SIZE)
    st.caption(
        f"Showing {(page - 1) * PAGE_SIZE + 1}–"
        f"{min(page * PAGE_SIZE, total)} of {total} candidates."
    )

    for candidate in candidates:
        with st.container(border=True):
            details, evidence = st.columns([1, 1])
            with details:
                st.markdown(f"**Candidate #{candidate['id']}**")
                st.write(f"Status: {candidate['review_status']}")
                st.write(f"Confidence: {candidate['confidence']:.1%}")
                st.write(f"Source: {Path(candidate['source_path']).name}")
                if candidate["frame_number"] is not None:
                    st.write(f"Frame: {candidate['frame_number']}")
                if candidate["track_id"]:
                    st.write(f"Track ID: {candidate['track_id']}")
                st.caption(f"Detected at: {candidate['created_at']}")

                if status == "pending_review":
                    confirm, reject = st.columns(2)
                    if confirm.button(
                        "Confirm",
                        key=f"confirm_{candidate['id']}",
                        width="stretch",
                    ):
                        try:
                            update_review_status(candidate["id"], "confirmed")
                            st.toast(f"Candidate #{candidate['id']} confirmed.")
                            st.rerun()
                        except ValueError as error:
                            st.error(str(error))
                    if reject.button(
                        "Reject",
                        key=f"reject_{candidate['id']}",
                        width="stretch",
                    ):
                        try:
                            update_review_status(candidate["id"], "rejected")
                            st.toast(f"Candidate #{candidate['id']} rejected.")
                            st.rerun()
                        except ValueError as error:
                            st.error(str(error))

            with evidence:
                evidence_path = Path(candidate["evidence_image_path"])
                if evidence_path.is_file():
                    st.image(str(evidence_path), caption="Detection evidence", width="stretch")
                else:
                    st.warning(f"Evidence image is missing: {evidence_path}")


def show_challan_form():
    st.subheader("Match a rider and create a challan")
    st.warning(
        "Demo only: upload a clear image containing the detected rider's face. "
        "Verify the suggested identity yourself. This dashboard records a simulated "
        "SMS; it does not send real messages or issue a legal notice."
    )
    st.caption(
        "The system first tries the selected detection evidence. If it finds no face "
        "or more than one face, choose “Upload a clear face crop”."
    )
    candidates = get_challan_candidates()
    if not candidates:
        st.info("Confirm a detection in the Review candidates tab before continuing.")
        return

    labels = {
        row["id"]: (
            f"Candidate #{row['id']} — {Path(row['source_path']).name}, "
            f"confidence {row['confidence']:.1%}"
        )
        for row in candidates
    }
    detection_id = st.selectbox(
        "Confirmed detection",
        options=list(labels),
        format_func=lambda value: labels[value],
    )
    evidence_by_id = {row["id"]: row for row in candidates}
    detection = evidence_by_id[detection_id]
    evidence_path = Path(detection["evidence_image_path"])
    if evidence_path.is_file():
        st.image(
            str(evidence_path),
            caption="Detection evidence — verify this is the correct rider",
            width="stretch",
        )
    else:
        st.error(f"Detection evidence is missing: {evidence_path}")

    face_source = st.radio(
        "Face image",
        ("Use detection evidence", "Upload a clear face crop"),
        horizontal=True,
        key="challan_face_source",
    )
    face_file = None
    if face_source == "Upload a clear face crop":
        face_file = st.file_uploader(
            "Rider face crop",
            type=("jpg", "jpeg", "png", "webp"),
            help="Use one clear face crop from the selected detection evidence.",
            key="challan_face_crop",
        )

    with st.form("match_rider_form"):
        location = st.text_input("Violation location")
        st.write(f"Fixed demo fine: **PKR {FINE_AMOUNT_PKR:,}**")
        payment = st.text_input(
            "Payment instructions",
            placeholder="Enter the instructions used in your project demo.",
        )
        appeal = st.text_input(
            "Appeal instructions",
            placeholder="Enter the instructions used in your project demo.",
        )
        submitted = st.form_submit_button(
            "Find face match",
            type="primary",
        )

    if submitted:
        st.session_state.pop("challan_match", None)
        if face_source == "Upload a clear face crop" and face_file is None:
            st.error("Choose a face photo before finding a match.")
        else:
            try:
                if face_source == "Upload a clear face crop":
                    uploaded_image = Image.open(
                        BytesIO(face_file.getvalue())
                    ).convert("RGB")
                    face_image = uploaded_image
                else:
                    face_image = evidence_path
                with st.spinner("Comparing the face with consented database records..."):
                    matched_cnic, distance, alternatives = match_person_image(
                        face_image,
                        threshold=MATCH_THRESHOLD,
                    )
                st.session_state["challan_form_values"] = {
                    "detection_id": detection_id,
                    "location": location,
                    "payment": payment,
                    "appeal": appeal,
                }
                if matched_cnic is None:
                    st.warning(
                        f"No confident match was found (best distance {distance:.3f}). "
                        "Do not create a challan for an unknown or ambiguous face."
                    )
                    for other_distance, _other_cnic, other_name in alternatives:
                        st.write(f"{other_name}: distance {other_distance:.3f}")
                else:
                    person = get_consented_person(matched_cnic)
                    st.session_state["challan_match"] = {
                        "detection_id": detection_id,
                        "cnic": matched_cnic,
                        "name": person["name"],
                        "phone": person["phone"],
                        "distance": distance,
                    }
            except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
                st.error(f"Face matching could not be completed: {error}")

    match = st.session_state.get("challan_match")
    saved_form = st.session_state.get("challan_form_values")
    if match and saved_form and match["detection_id"] == detection_id:
        st.markdown("#### Suggested identity")
        st.write(f"Name: {match['name']}")
        st.write(f"CNIC: {match['cnic']}")
        st.write(f"Registered phone: {match['phone']}")
        st.write(f"Face distance: {match['distance']:.3f}")
        st.caption(
            "A lower distance indicates a closer embedding match. A face match is "
            f"only a suggestion and must be checked against the evidence. "
            f"Maximum distance: {MATCH_THRESHOLD:.2f}; ambiguous matches are refused."
        )
        verified = st.checkbox(
            "I reviewed the evidence and verified this is the rider in the selected detection.",
            key=f"verify_rider_{detection_id}",
        )
        if st.button(
            "Create challan and save simulated SMS",
            type="primary",
            disabled=not verified,
            key=f"create_challan_{detection_id}",
        ):
            try:
                result = create_challan(
                    detection_id=detection_id,
                    cnic=match["cnic"],
                    location=saved_form["location"],
                    payment_instructions=saved_form["payment"],
                    appeal_instructions=saved_form["appeal"],
                )
                st.session_state.pop("challan_match", None)
                st.session_state.pop("challan_form_values", None)
                st.success(
                    f"Demo challan #{result['challan_id']} saved for {result['name']}. "
                    f"A simulated SMS record was saved for {result['phone']}."
                )
                st.text_area(
                    "Bilingual challan message",
                    value=result["message"],
                    height=320,
                    key=f"challan_message_{result['challan_id']}",
                )
            except (
                FileNotFoundError,
                ValueError,
                OSError,
                RuntimeError,
                sqlite3.IntegrityError,
            ) as error:
                st.error(f"Challan could not be created: {error}")


def main():
    st.title("Helmet Violation Review")
    st.write(
        "Run the helmet detector and review its possible no-helmet detections. "
        "Confirm evidence manually, match a rider face, and verify the suggested "
        "identity before creating a demo challan."
    )
    st.info(
        "Identity matching is a suggestion, not proof. A challan is saved only "
        "after human review and identity confirmation. Notifications are simulated."
    )

    if not DEFAULT_WEIGHTS.is_file():
        st.error(f"YOLO weights were not found: {DEFAULT_WEIGHTS}")
        st.stop()

    initialize_database()
    detection_tab, review_tab, challan_tab = st.tabs(
        ("Run detection", "Review candidates", "Create challan")
    )
    with detection_tab:
        show_detection_form()
    with review_tab:
        show_candidate_review()
    with challan_tab:
        show_challan_form()

    st.caption(f"Local SQLite database: {DATABASE_PATH}")


if __name__ == "__main__":
    main()
