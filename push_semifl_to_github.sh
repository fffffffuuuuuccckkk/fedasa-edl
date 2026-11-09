#!/usr/bin/env bash
set -Eeuo pipefail

# Safe default: preview files and Git changes; do not push.
# Preview: bash push_semifl_to_github.sh
# Push:    PUSH=true bash push_semifl_to_github.sh

OWNER="${OWNER:-fffffffuuuuuccckkk}"
REPO_NAME="${REPO_NAME:-fedasa-edl}"
SOURCE_DIR="${SOURCE_DIR:-/data/OuXiaoyu/semifl_original/SemiFL}"
BRANCH="${BRANCH:-main}"
REMOTE_URL="${REMOTE_URL:-git@github.com:${OWNER}/${REPO_NAME}.git}"
WORK_DIR="${WORK_DIR:-/tmp/${REPO_NAME}-sanitized-push}"
CLONE_DIR="${CLONE_DIR:-/tmp/${REPO_NAME}-git-work}"
PUSH="${PUSH:-false}"
MAX_FILE_SIZE_MB="${MAX_FILE_SIZE_MB:-50}"
SSH_TRANSPORT="${SSH_TRANSPORT:-${SOURCE_DIR}/.push-tools/git_ssh_paramiko.py}"

log() { printf '[push-semifl] %s\n' "$*"; }
die() { log "ERROR: $*"; exit 2; }

for cmd in rsync git find sort du awk; do
  command -v "$cmd" >/dev/null 2>&1 || die "Missing command: $cmd"
done

[[ -d "$SOURCE_DIR" ]] || die "Source directory not found: $SOURCE_DIR"
[[ "$PUSH" == "true" || "$PUSH" == "false" ]] || die "PUSH must be true or false"
[[ "$MAX_FILE_SIZE_MB" =~ ^[1-9][0-9]*$ ]] || die "MAX_FILE_SIZE_MB must be a positive integer"

# Temp paths are deliberately restricted: the script removes them before use.
[[ "$WORK_DIR" =~ ^/tmp/[A-Za-z0-9_.-]+-sanitized-push$ ]] || die "Unsafe WORK_DIR: $WORK_DIR"
[[ "$CLONE_DIR" =~ ^/tmp/[A-Za-z0-9_.-]+-git-work$ ]] || die "Unsafe CLONE_DIR: $CLONE_DIR"
[[ "$WORK_DIR" != "$CLONE_DIR" ]] || die "WORK_DIR and CLONE_DIR must differ"

# Optional Paramiko SSH adapter, as in your FPEM script.
if [[ ! -x "$SSH_TRANSPORT" && -x /data/OuXiaoyu/STEVE_CODE/.push-tools/git_ssh_paramiko.py ]]; then
  SSH_TRANSPORT=/data/OuXiaoyu/STEVE_CODE/.push-tools/git_ssh_paramiko.py
fi

git_remote() {
  if [[ -x "$SSH_TRANSPORT" ]]; then
    PYTHONWARNINGS=ignore GIT_SSH="$SSH_TRANSPORT" GIT_SSH_VARIANT=ssh "$@"
  else
    "$@"
  fi
}

log "Creating clean publication copy at $WORK_DIR"
rm -rf -- "$WORK_DIR"
mkdir -p -- "$WORK_DIR"

# Exclusions are relative to SOURCE_DIR. The two large directories never
# enter the temporary copy, regardless of the source .gitignore.
rsync -a \
  --exclude='/.git/' \
  --exclude='**/.git/' \
  --exclude='/.push-tools/' \
  --exclude='/src/data/' \
  --exclude='/src/output/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache/' \
  --exclude='.mypy_cache/' \
  --exclude='.DS_Store' \
  --exclude='.env' \
  --exclude='.env.*' \
  --exclude='*.pem' \
  --exclude='*.key' \
  "$SOURCE_DIR/" "$WORK_DIR/"

# Keep future commits from accidentally tracking these paths.
[[ ! -L "$WORK_DIR/.gitignore" ]] || die "Refusing symlinked .gitignore"
cat >> "$WORK_DIR/.gitignore" <<'EOF'

# SemiFL GitHub publication policy
/src/data/
/src/output/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.env
.env.*
*.pem
*.key
EOF

for excluded in src/data src/output .git; do
  [[ ! -e "$WORK_DIR/$excluded" && ! -L "$WORK_DIR/$excluded" ]] \
    || die "Excluded path still exists: $excluded"
done

# Fail rather than silently upload unexpected large files. Adjust if needed.
oversized="$(find "$WORK_DIR" -type f -size "+${MAX_FILE_SIZE_MB}M" -print)"
if [[ -n "$oversized" ]]; then
  printf '%s\n' "$oversized"
  die "Files exceed ${MAX_FILE_SIZE_MB} MiB. Exclude them or adjust MAX_FILE_SIZE_MB."
fi

log "Publication copy: $(find "$WORK_DIR" -type f | wc -l | tr -d ' ') files, $(du -sh "$WORK_DIR" | awk '{print $1}')"
log "Excluded: src/data/ and src/output/"
log "Largest files in snapshot (bytes path):"
find "$WORK_DIR" -type f -printf '%s %P\n' | sort -nr | head -15 || true

# Fetch into a separate temporary Git working tree so the original tree,
# .git history, and source files remain untouched.
log "Checking remote: $REMOTE_URL"
git_remote git ls-remote "$REMOTE_URL" >/dev/null \
  || die "Cannot access GitHub repository. Check SSH / Paramiko credentials."

remote_head="$(git_remote git ls-remote "$REMOTE_URL" "refs/heads/$BRANCH" | awk 'NR==1 {print $1}')"

rm -rf -- "$CLONE_DIR"
if [[ -n "$remote_head" ]]; then
  git_remote git clone -q --single-branch --branch "$BRANCH" "$REMOTE_URL" "$CLONE_DIR"
else
  mkdir -p -- "$CLONE_DIR"
  git -C "$CLONE_DIR" init -q
  git -C "$CLONE_DIR" checkout -q -b "$BRANCH"
  git -C "$CLONE_DIR" remote add origin "$REMOTE_URL"
fi

# Sync exact sanitized snapshot. This can delete stale files from the remote
# branch, but it NEVER deletes files from SOURCE_DIR.
# The .git directory in CLONE_DIR is protected from rsync deletion.
rsync -a --delete --exclude='/.git/' "$WORK_DIR/" "$CLONE_DIR/"
git -C "$CLONE_DIR" add -A

log "Staged changes (A=add, M=modify, D=delete):"
git -C "$CLONE_DIR" status --short

if [[ "$PUSH" != "true" ]]; then
  log "PREVIEW ONLY: GitHub has NOT been modified."
  log "Review snapshot: $WORK_DIR"
  log "Review staged Git diff: git -C '$CLONE_DIR' diff --cached --stat"
  log "To publish: PUSH=true bash '$SOURCE_DIR/push_semifl_to_github.sh'"
  exit 0
fi

if git -C "$CLONE_DIR" diff --cached --quiet; then
  log "No changes detected; nothing to push."
  exit 0
fi

# Keep real Git history on existing branches, without force pushes.
git -C "$CLONE_DIR" config user.name \
  "${GIT_AUTHOR_NAME:-$(git config --global user.name 2>/dev/null || echo OuXiaoyu)}"
git -C "$CLONE_DIR" config user.email \
  "${GIT_AUTHOR_EMAIL:-$(git config --global user.email 2>/dev/null || echo 'OuXiaoyu@users.noreply.github.com')}"

git -C "$CLONE_DIR" commit -q -m "${COMMIT_MESSAGE:-Publish SemiFL source (without data and output)}"
log "Pushing branch $BRANCH (no force push)"
git_remote git -C "$CLONE_DIR" push -u origin "$BRANCH"
log "Done: https://github.com/${OWNER}/${REPO_NAME}"
