#!/bin/bash

# This script runs from the Fed-Fold directory
# It processes folders in ./data/pdb_recent/fasta_files

# Set the base directory path (relative to Fed-Fold)
BASE_DIR="./data/pdb_recent/fasta_files"
OPENFOLD_SCRIPT="./run_pretrained_openfold.py"

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR does not exist"
    echo "Current working directory: $(pwd)"
    exit 1
fi

# Check if OpenFold script exists
if [ ! -f "$OPENFOLD_SCRIPT" ]; then
    echo "Error: OpenFold script $OPENFOLD_SCRIPT not found"
    echo "Current working directory: $(pwd)"
    exit 1
fi

echo "========================================"
echo "OpenFold Batch Processor"
echo "Running from: $(pwd)"
echo "Processing folders in: $BASE_DIR"
echo "========================================"

# Counter for processed folders
count=0
success=0
failed=0
skipped=0

# Loop through each folder in the base directory
for folder in "$BASE_DIR"/*/; do
    # Check if it's actually a directory
    [ -d "$folder" ] || continue
    
    # Get folder name without path
    folder_name=$(basename "$folder")
    
    echo "[$((count+1))] Processing folder: $folder_name"
    
    # Find .fasta files in the folder
    # Use full path to the folder
    fasta_file=("$BASE_DIR/$folder_name"/*.fasta)
    
    # Check if there are any fasta files
    if [ ${#fasta_file[@]} -eq 0 ] || [ ! -f "${fasta_file[0]}" ]; then
        echo "  ⚠️  Warning: No .fasta files found in $folder_name, skipping..."
        ((skipped++))
        ((count++))
        continue
    fi
    
    # Process each fasta file in the folder
    if [ -f "$fasta_file" ]; then
        echo "  Found fasta: $(basename "$fasta_file")"
        echo "  Running OpenFold..."
        
        # Run OpenFold command with the full path to the fasta file
        python "$OPENFOLD_SCRIPT" "$BASE_DIR/$folder_name" ./data/pdb_recent/mmcif_files --use_precomputed_alignments ./data/pdb_recent/embeddings_output_dir --output_dir ./data/pdb_recent/soloseq_inference_outputs --model_device "cuda:0" --config_preset "seq_model_esm1b_ptm" --openfold_checkpoint_path openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt
        
        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "  ✅ Successfully processed $folder_name/$(basename "$fasta_file")"
            ((success++))
        else
            echo "  ❌ Failed to process $folder_name/$(basename "$fasta_file")"
            ((failed++))
        fi
    fi
    
    echo "----------------------------------------"
    ((count++))
done

echo "========================================"
echo "Processing complete!"
echo "Total folders processed: $count"
echo "Files successfully processed: $success"
echo "Files failed: $failed"
echo "Folders skipped (no fasta): $skipped"
echo "========================================"