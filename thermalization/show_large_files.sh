git rev-list --objects --all | sort -k 2 | while read -r obj file; do
  echo "$(git cat-file -s "$obj") $file"
done | sort -nr | head -20

