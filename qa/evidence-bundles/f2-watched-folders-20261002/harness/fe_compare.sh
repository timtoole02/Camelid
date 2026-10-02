#!/usr/bin/env bash
# Is the project-context smoke failure in the frontend job specific to the folder branch?
# Run the same job, with the same environment, on the branch below it (#788) and on the folder branch
# with PUPPETEER_EXECUTABLE_PATH unset (as in the first, passing run).
CHROME=$(ls -d ~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome | head -1)
cd /tmp/f2r/wt-library && git log --oneline -1
CHROME_PATH="$CHROME" PUPPETEER_EXECUTABLE_PATH="$CHROME" REPO=/tmp/f2r/wt-library python3 /tmp/f2r/ci_job_wt.py frontend > /tmp/f2r/fe-788.log 2>&1
echo "#788 with PUPPETEER_EXECUTABLE_PATH:"; grep -E "^FAIL|JOB_DONE" /tmp/f2r/fe-788.log
cd /tmp/f2r/wt-folders && git log --oneline -1
env -u PUPPETEER_EXECUTABLE_PATH REPO=/tmp/f2r/wt-folders python3 /tmp/f2r/ci_job_wt.py frontend > /tmp/f2r/fe-fw-nopp.log 2>&1
echo "folders without PUPPETEER_EXECUTABLE_PATH:"; grep -E "^FAIL|JOB_DONE" /tmp/f2r/fe-fw-nopp.log
echo FE_COMPARE_DONE
