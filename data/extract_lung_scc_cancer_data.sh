#!/bin/bash

# Define the input file containing Lung-scc-related patient IDs
input_file="pcawg_primary_type.csv"

# Define the output files
muts_output="lung_scc_muts_final.txt"
sig_output="lung_scc_sig_final.txt"

# Clear the output files if they already exist
> "$muts_output"
> "$sig_output"

# Initialize a flag to check the consistency of line counts
line_count_match=true

# Extract the patient IDs related to Lung-SCC cancer
grep -i "Lung-SCC" "$input_file" | while IFS=',' read -r patient_id cancer_type; do
  # Remove quotes from patient_id
  patient_id=$(echo "$patient_id" | tr -d '"')
  
  # Check if the corresponding files exist in the muts folder and concatenate their contents
  muts_file="muts/${patient_id}.txt"
  if [ -f "$muts_file" ]; then
    # Remove empty lines and append contents to the output file
    grep -v '^$' "$muts_file" >> "$muts_output"
  else
    echo "Warning: $muts_file not found." >&2
  fi
  
  # Check if the corresponding files exist in the sig folder and concatenate their contents
  sig_file="sig/${patient_id}.txt"
  if [ -f "$sig_file" ]; then
    # Add "SBS" label if it doesn't exist and then remove empty lines
    if ! grep -q '^SBS' "$sig_file"; then
      echo "SBS" > temp_sig_file.txt
      grep -v '^$' "$sig_file" >> temp_sig_file.txt
      cat temp_sig_file.txt >> "$sig_output"
      rm temp_sig_file.txt
    else
      grep -v '^$' "$sig_file" >> "$sig_output"
    fi
  else
    echo "Warning: $sig_file not found." >&2
  fi
done

# Count the number of lines in the muts and sig output files
muts_lines=$(grep -cv '^$' "$muts_output")  # Count non-empty lines
sig_lines=$(grep -cv '^$' "$sig_output")    # Count non-empty lines

# Exclude the first line in the muts output as it is a label
muts_lines=$((muts_lines - 1))

# Compare the number of lines and set the flag accordingly
if [ "$muts_lines" -ne "$sig_lines" ]; then
  line_count_match=false
fi

# Print the result of the line count comparison
if $line_count_match; then
  echo "The number of lines in the muts file matches the number of lines in the sig file."
else
  echo "The number of lines in the muts file does NOT match the number of lines in the sig file."
fi

echo "Combination of files complete."
