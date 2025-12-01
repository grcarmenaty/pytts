#!/usr/bin/env python3
"""
Generate PDF report from an aggregated CSV file.

Usage:
    python generate_pdf_from_csv.py <csv_file> [output_pdf]

Examples:
    python generate_pdf_from_csv.py modernisme_logs/all_games_20251201_100116.csv
    python generate_pdf_from_csv.py modernisme_logs/all_games_20251201_100116.csv custom_report.pdf
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from generate_report import generate_pdf_report


def main():
    """Generate PDF report from CSV file."""
    if len(sys.argv) < 2:
        print("Error: CSV file path required")
        print("\nUsage:")
        print("  python generate_pdf_from_csv.py <csv_file> [output_pdf]")
        print("\nExamples:")
        print("  python generate_pdf_from_csv.py modernisme_logs/all_games_20251201_100116.csv")
        print("  python generate_pdf_from_csv.py modernisme_logs/all_games_20251201_100116.csv custom_report.pdf")
        sys.exit(1)

    csv_file = sys.argv[1]
    csv_path = Path(csv_file)

    # Check if CSV file exists
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_file}")
        sys.exit(1)

    # Determine output PDF path
    if len(sys.argv) > 2:
        pdf_file = sys.argv[2]
        pdf_path = Path(pdf_file)
    else:
        # Generate PDF name based on CSV name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_name = f"{csv_path.stem}_report_{timestamp}.pdf"
        pdf_path = csv_path.parent / pdf_name

    print("=" * 70)
    print("MODERNISME PDF REPORT GENERATOR")
    print("=" * 70)
    print(f"\nInput CSV:  {csv_path}")
    print(f"Output PDF: {pdf_path}")
    print("\nGenerating report...")

    try:
        generate_pdf_report(str(csv_path), str(pdf_path))
        print("\n" + "=" * 70)
        print("✓ PDF REPORT GENERATED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nReport saved to: {pdf_path}")
        print(f"File size: {pdf_path.stat().st_size / 1024 / 1024:.1f} MB")
    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ ERROR GENERATING PDF REPORT")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
