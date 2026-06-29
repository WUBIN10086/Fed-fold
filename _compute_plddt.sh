#!/bin/bash

# Set the directory containing the PDB files
PDB_DIR="./data/pdb_recent/soloseq_inference_outputs/predictions"

# Check if directory exists
if [ ! -d "$PDB_DIR" ]; then
    echo "Error: Directory $PDB_DIR does not exist"
    exit 1
fi

# Process each file ending with unrelaxed.pdb
for pdb_file in "$PDB_DIR"/*unrelaxed.pdb; do
    # Check if any files exist (prevents processing if no files match)
    if [ -f "$pdb_file" ]; then
        echo "Processing: $pdb_file"
        python scripts/plddt_from_pdb.py "$pdb_file"
    else
        echo "No unrelaxed.pdb files found in $PDB_DIR"
        break
    fi
done

echo "All files processed"