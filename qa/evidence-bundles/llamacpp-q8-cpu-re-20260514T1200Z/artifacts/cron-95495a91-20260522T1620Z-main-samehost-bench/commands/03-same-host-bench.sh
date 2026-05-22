export PATH=/usr/local/bin:/home/ubuntu/.cargo/bin:$PATH
export CARGO_TARGET_DIR=/home/ubuntu/work/camelid-targets/backend-95495a91
cd /home/ubuntu/work/camelid-cron-95495a91-20260522T161257Z-main
node scripts/bench-llama3-same-host.mjs --model /home/ubuntu/models/Llama-3.2-3B-Instruct-Q8_0.gguf --backend-bin /home/ubuntu/work/camelid-targets/backend-95495a91/release/camelid --llama-server /home/ubuntu/work/llama.cpp-clean-20260517/build/bin/llama-server --out /home/ubuntu/work/camelid-cron-95495a91-20260522T161257Z-main/qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260522T1620Z-main-samehost-bench/same-host-bench.json --repeats 2 --warmup 0 --max-tokens 8 --threads 8 --require-marker --unique-prompt
