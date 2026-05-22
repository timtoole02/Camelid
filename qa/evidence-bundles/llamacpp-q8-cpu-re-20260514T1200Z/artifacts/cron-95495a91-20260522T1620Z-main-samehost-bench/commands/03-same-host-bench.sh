export PATH=/usr/local/bin:$UBUNTU_HOME/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$UBUNTU_HOME/work/camelid-targets/backend-95495a91
cd $UBUNTU_HOME/work/camelid-cron-95495a91-20260522T161257Z-main
node scripts/bench-llama3-same-host.mjs --model $UBUNTU_HOME/models/Llama-3.2-3B-Instruct-Q8_0.gguf --backend-bin $UBUNTU_HOME/work/camelid-targets/backend-95495a91/release/camelid --llama-server $UBUNTU_HOME/work/llama.cpp-clean-20260517/build/bin/llama-server --out $UBUNTU_HOME/work/camelid-cron-95495a91-20260522T161257Z-main/qa/evidence-bundles/llamacpp-q8-cpu-re-20260514T1200Z/artifacts/cron-95495a91-20260522T1620Z-main-samehost-bench/same-host-bench.json --repeats 2 --warmup 0 --max-tokens 8 --threads 8 --require-marker --unique-prompt
