#!/bin/zsh
setopt no_unset pipe_fail
run_dir=$1
operator_home=${CAMELID_OPERATOR_HOME:-${HOME:?set CAMELID_OPERATOR_HOME or HOME}}
model_root=${CAMELID_MODEL_ROOT:-$operator_home/models}
repo_candidate=${CAMELID_REPO_ROOT:-${0:A:h}/../../..}
repo=${repo_candidate:A}
bundle=${repo}/qa/evidence-bundles/gemma4-26b-mtp-assistant-oracle
wd=${bundle}/run_load_only_watchdog.py
adm=${run_dir}/camelid-mtp-admission
exp=${run_dir}/gemma4-mtp-assistant-experiment
assistant=${CAMELID_GEMMA4_MTP_ASSISTANT_PATH:-$model_root/gemma4-26b-a4b-mtp-qat-assistant/model.safetensors}
stage=${CAMELID_GEMMA4_MTP_STAGE_ORACLE_JSON:-$model_root/gemma4-mtp-pair/runs/mtp-nim.fPTehuPt/stage-oracle.json}
cam_lock=${CAMELID_CAM_LOCK:-$operator_home/bin/cam-lock.sh}
ev=${run_dir}/native-admission.json
nonce=$(/usr/bin/uuidgen | /usr/bin/tr -d '-' | /usr/bin/tr 'A-Z' 'a-z')$(/usr/bin/uuidgen | /usr/bin/tr -d '-' | /usr/bin/tr 'A-Z' 'a-z' | /usr/bin/cut -c1-16)
print -r -- "${nonce}" > ${run_dir}/run-nonce.txt
cd ${run_dir}

print -r -- "== admission $(/bin/date -u +%FT%TZ)"
CAM_SESSION_PID=$$ $cam_lock /usr/bin/env -i \
  HOME=$operator_home PATH=/usr/bin:/bin:/usr/sbin:/sbin TMPDIR=/tmp \
  CAMELID_GEMMA4_MTP_NATIVE_ASSISTANT_PATH=${assistant} \
  CAMELID_GEMMA4_MTP_RECURRENCE_ORACLE_JSON=${bundle}/assistant_recurrence7_bf16_cpu.json \
  CAMELID_GEMMA4_MTP_STAGE_ORACLE_JSON=${stage} \
  CAMELID_GEMMA4_MTP_RECURRENCE_GENERATION_RECEIPT_JSON=${bundle}/recurrence7_generation_receipt.json \
  CAMELID_GEMMA4_MTP_NATIVE_ADMISSION_EVIDENCE_JSON=${ev} \
  CAMELID_GEMMA4_MTP_NATIVE_ADMISSION_RUN_NONCE=${nonce} \
  /usr/bin/python3 ${wd} --report ${ev} \
    --watchdog-log ${run_dir}/admission-watchdog.jsonl --child-log ${run_dir}/admission.log \
    -- ${adm} metal::gemma4_mtp::tests::official_target_free_bf16_oracle_matches_native_seven_proposal_recurrence \
       --exact --ignored --nocapture --test-threads=1
arc=$?
print -r -- "== admission rc=${arc}"
(( arc == 0 )) || exit ${arc}

print -r -- "== pilot (OBSERVABILITY ONLY: +CAMELID_GEMMA4_GHOST_METAL_TIMING=1) $(/bin/date -u +%FT%TZ)"
CAM_SESSION_PID=$$ $cam_lock /usr/bin/env -i \
  HOME=$operator_home PATH=/usr/bin:/bin:/usr/sbin:/sbin TMPDIR=/tmp \
  CAMELID_GEMMA4_GHOST_METAL_TIMING=1 \
  CAMELID_GEMMA4_MTP_EXPERIMENT=1 \
  CAMELID_GEMMA4_MTP_ASSISTANT_PATH=${assistant} \
  CAMELID_GEMMA4_MTP_NATIVE_ADMISSION_EVIDENCE_JSON=${ev} \
  CAMELID_GEMMA4_MTP_NATIVE_ADMISSION_RUN_NONCE=${nonce} \
  CAMELID_GEMMA4_MTP_NATIVE_ADMISSION_TEST_EXE=${adm} \
  CAMELID_GEMMA4_MTP_PILOT_ONLY=1 \
  CAMELID_GEMMA4_MTP_REPORT_PATH=${run_dir}/nim-pilot-report.json \
  /usr/bin/python3 ${wd} --report ${run_dir}/nim-pilot-report.json \
    --watchdog-log ${run_dir}/nim-watchdog.jsonl --child-log ${run_dir}/nim-pilot.log \
    -- ${exp} gemma4_mtp_assistant_experiment --exact --ignored --nocapture --test-threads=1
prc=$?
print -r -- "== pilot rc=${prc} (86=watchdog abort, expected in lane I) $(/bin/date -u +%FT%TZ)"
exit 0
