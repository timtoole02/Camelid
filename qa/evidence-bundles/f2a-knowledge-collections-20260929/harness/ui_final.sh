#!/usr/bin/env bash
# After the live evidence: the UI smokes and the mutation check on the committed tree.
until grep -q EVIDENCE_DONE /tmp/f2c/evidence.log 2>/dev/null; do sleep 10; done
cd ~/Camelid || exit 1
echo "commit=$(git rev-parse --short HEAD) tree_clean=$([ -z "$(git status --porcelain --untracked-files=no)" ] && echo yes || echo no)"
rm -rf /tmp/f2c/front   # run_smokes.sh copies nothing when the staging dir is absent
mkdir -p /tmp/f2c/front
bash /tmp/f2c/run_smokes.sh
python3 /tmp/f2c/mutate_collections.py
echo "tree after: $(git status --porcelain --untracked-files=no | wc -l) tracked changes"
echo UI_FINAL_DONE
