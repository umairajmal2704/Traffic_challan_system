"""Generate simple English and Urdu traffic-challan messages."""

import argparse
import sys
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path


VIOLATIONS = {
    "no_helmet": {
        "english": "Riding a motorcycle without a helmet",
        "urdu": "موٹر سائیکل چلاتے وقت ہیلمٹ نہ پہننا",
    },
}


def validate_timestamp(value):
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            "Timestamp must be ISO format, for example 2026-10-16T14:30:00."
        ) from error
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def validate_fine(value):
    try:
        fine = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("Fine must be a valid amount in PKR.") from error
    if not fine.is_finite() or fine < 0:
        raise ValueError("Fine must be a finite amount greater than or equal to zero.")
    return fine.quantize(Decimal("0.01"))


def generate_messages(
    violation_type,
    timestamp,
    location,
    fine,
    plate_no="",
    payment="",
    appeal="",
    person_name="",
):
    """Fill bilingual message templates with the supplied challan details."""
    if violation_type not in VIOLATIONS:
        raise ValueError(f"Unsupported violation type: {violation_type}")

    timestamp = validate_timestamp(timestamp)
    fine = validate_fine(str(fine))
    fields = {
        "violation_en": VIOLATIONS[violation_type]["english"],
        "violation_ur": VIOLATIONS[violation_type]["urdu"],
        "timestamp": timestamp,
        "location": location.strip(),
        "fine": f"{fine:.2f}",
        "plate_no": plate_no.strip(),
        "payment": payment.strip(),
        "appeal": appeal.strip(),
        "person_name": person_name.strip(),
    }
    if not all(fields[key] for key in ("location", "payment", "appeal")):
        raise ValueError("Location, payment instructions, and appeal instructions are required.")

    english_lines = ["DEMO TRAFFIC CHALLAN — NOT AN OFFICIAL NOTICE"]
    if fields["person_name"]:
        english_lines.append(f"To: {fields['person_name']}")
    english_lines.extend(
        [
            f"Violation: {fields['violation_en']}",
            f"Date and time: {fields['timestamp']}",
            f"Location: {fields['location']}",
            f"Fine: PKR {fields['fine']}",
        ]
    )
    if fields["plate_no"]:
        english_lines.append(
            f"Vehicle plate (evidence only): {fields['plate_no']}"
        )
    english_lines.extend(
        [
            f"How to pay: {fields['payment']}",
            f"How to appeal: {fields['appeal']}",
            "Please verify the notice and evidence before making a payment.",
        ]
    )

    urdu_lines = ["ڈیمو ٹریفک چالان — سرکاری نوٹس نہیں"]
    if fields["person_name"]:
        urdu_lines.append(f"نام: {fields['person_name']}")
    urdu_lines.extend(
        [
            f"خلاف ورزی: {fields['violation_ur']}",
            f"تاریخ اور وقت: {fields['timestamp']}",
            f"مقام: {fields['location']}",
            f"جرمانہ: {fields['fine']} پاکستانی روپے",
        ]
    )
    if fields["plate_no"]:
        urdu_lines.append(
            f"گاڑی کا نمبر (صرف ثبوت کے لیے): {fields['plate_no']}"
        )
    urdu_lines.extend(
        [
            f"ادائیگی کا طریقہ: {fields['payment']}",
            f"اپیل کرنے کا طریقہ: {fields['appeal']}",
            "ادائیگی سے پہلے نوٹس اور ثبوت کی تصدیق کریں۔",
        ]
    )
    english = "\n".join(english_lines)
    urdu = "\n".join(urdu_lines)
    return {"english": english, "urdu": urdu}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Generate English and Urdu challan messages from templates."
    )
    parser.add_argument("--violation", choices=sorted(VIOLATIONS), required=True)
    parser.add_argument(
        "--timestamp",
        required=True,
        help="ISO date/time, for example 2026-10-16T14:30:00.",
    )
    parser.add_argument("--location", required=True)
    parser.add_argument("--fine", required=True, help="Fine amount in Pakistani rupees.")
    parser.add_argument(
        "--plate",
        default="",
        help="Optional plate evidence; the plate is not used to choose the recipient.",
    )
    parser.add_argument("--payment", required=True, help="Payment instructions to include.")
    parser.add_argument("--appeal", required=True, help="Appeal instructions to include.")
    parser.add_argument(
        "--output",
        help="Optional UTF-8 text-file path for saving both language versions.",
    )
    args = parser.parse_args()

    try:
        messages = generate_messages(
            args.violation,
            args.timestamp,
            args.location,
            args.fine,
            args.plate,
            args.payment,
            args.appeal,
        )
        output = (
            f"--- ENGLISH ---\n{messages['english']}\n\n"
            f"--- URDU ---\n{messages['urdu']}\n"
        )
        print(output)
        if args.output:
            output_path = Path(args.output).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output, encoding="utf-8")
            print(f"Messages saved to: {output_path}")
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
