# add large files to gitignore 
# Usage: bash add_large.sh
#!/bin/bash

# Define the size threshold (e.g., 10M for 10 MB)
SIZE_THRESHOLD="+50M"

# Find all files larger than the threshold and add them to .gitignore
find . -type f -size $SIZE_THRESHOLD | while read -r file; do
  # Remove the ./ prefix from the file path
  file=${file#./}
  # Check if the file is already in .gitignore
  if ! grep -qx "$file" .gitignore; then
    echo "Adding $file to .gitignore"
    echo "$file" >> .gitignore
  fi
done

# Optional: Sort and remove duplicates in .gitignore
sort -uo .gitignore .gitignore

echo "Large files have been added to .gitignore."