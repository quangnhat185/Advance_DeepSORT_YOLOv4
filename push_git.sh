echo "Update comment: "
read line
var=$(git branch  --show-current)
echo "Current on branch ${var}"
git add .
git commit -m "${line}"
git push origin "${var}"