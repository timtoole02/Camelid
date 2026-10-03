#!/usr/bin/env bash
# After the live evidence: the UI smokes and the mutation check on the committed tree.
until grep -q EVIDENCE_DONE /tmp/f2l/evidence2.log 2>/dev/null; do sleep 15; done
cd ~/Camelid || exit 1
echo "commit=$(git rev-parse --short HEAD) tree_clean=$([ -z "$(git status --porcelain --untracked-files=no)" ] && echo yes || echo no)"
bash /tmp/f2l/ui_check.sh mutate
echo "tree after: $(git status --porcelain --untracked-files=no | wc -l) tracked changes"
echo UI_FINAL_DONE
