import pandas as pd
import os


def convert_arff_to_csv(input_arff):
    """
    Converts a UCI ARFF dataset file to CSV format.

    Args:
        input_arff (str): Path to uploaded .arff file

    Returns:
        output_csv (str): Path to generated CSV file
    """

    if not os.path.exists(input_arff):
        raise FileNotFoundError(f"ARFF file not found: {input_arff}")

    # Output CSV path (same folder as input)
    output_csv = input_arff.replace(".arff", ".csv")

    # -------------------------------------------------------
    # Step 1: Read ARFF file
    # -------------------------------------------------------
    with open(input_arff, "r") as f:
        lines = f.readlines()

    # -------------------------------------------------------
    # Step 2: Extract attribute names
    # -------------------------------------------------------
    attributes = []
    data_start = 0

    for i, line in enumerate(lines):
        line = line.strip()

        if line.lower().startswith("@attribute"):
            attr_name = line.split()[1].strip("'\"")
            attributes.append(attr_name)

        elif line.lower().startswith("@data"):
            data_start = i + 1
            break

    # -------------------------------------------------------
    # Step 3: Extract data rows
    # -------------------------------------------------------
    data_rows = []

    for line in lines[data_start:]:
        line = line.strip()

        if not line or line.startswith("%"):
            continue

        if line.endswith(","):
            line = line[:-1]

        row = [item.strip() for item in line.split(",")]

        # Fix row length mismatch
        if len(row) < len(attributes):
            row += ["?"] * (len(attributes) - len(row))
        elif len(row) > len(attributes):
            row = row[:len(attributes)]

        data_rows.append(row)

    # -------------------------------------------------------
    # Step 4: Convert to DataFrame
    # -------------------------------------------------------
    df = pd.DataFrame(data_rows, columns=attributes)

    # -------------------------------------------------------
    # Step 5: Save CSV
    # -------------------------------------------------------
    df.to_csv(output_csv, index=False)

    return output_csv


# -------------------------------------------------------
# Optional standalone execution (for testing)
# -------------------------------------------------------
if __name__ == "__main__":
    test_input = "../data/raw/uci_ckd.arff"
    csv_path = convert_arff_to_csv(test_input)
    print(f"CSV saved at: {csv_path}")