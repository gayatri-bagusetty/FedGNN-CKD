import pandas as pd
import os

def convert_arff_to_csv(input_arff, output_csv):
    """
    Converts a UCI ARFF dataset file to a CSV format.
    """
    if not os.path.exists(input_arff):
        print(f"Error: The file {input_arff} does not exist.")
        return

    # Step 1: Read the ARFF file as text
    with open(input_arff, 'r') as f:
        lines = f.readlines()

    # Step 2: Extract attribute names
    attributes = []
    data_start = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if line.lower().startswith('@attribute'):
            # Attribute name is the second word
            attr_name = line.split()[1].strip("'\"")
            attributes.append(attr_name)
        elif line.lower().startswith('@data'):
            data_start = i + 1
            break

    # Step 3: Extract data rows
    data_rows = []
    for line in lines[data_start:]:
        line = line.strip()
        if not line or line.startswith('%'):  # Skip empty lines or comments
            continue
        
        # Remove trailing commas and split
        if line.endswith(','):
            line = line[:-1]
        row = [item.strip() for item in line.split(',')]
        
        # Handle rows with wrong length
        if len(row) < len(attributes):
            row += ['?'] * (len(attributes) - len(row))  # pad with '?'
        elif len(row) > len(attributes):
            row = row[:len(attributes)]  # truncate extra columns
        
        data_rows.append(row)

    # Step 4: Convert to DataFrame
    df = pd.DataFrame(data_rows, columns=attributes)

    # Step 5: Save as CSV
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Conversion completed! Saved as: {output_csv}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    # Update these paths as needed
    INPUT_FILE = "your_dataset.arff" 
    OUTPUT_FILE = "../data/raw/ckd_dataset.csv"
    
    convert_arff_to_csv(INPUT_FILE, OUTPUT_FILE)