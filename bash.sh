# List all remote branches except for 'main' (adjust as needed)
branches=$(git branch -r | grep -v "origin/main" | sed 's/origin\///')

# Iterate through branches and merge into 'main'
for branch in $branches; do
  git merge "origin/$branch" -m "Merge $branch into main"
done