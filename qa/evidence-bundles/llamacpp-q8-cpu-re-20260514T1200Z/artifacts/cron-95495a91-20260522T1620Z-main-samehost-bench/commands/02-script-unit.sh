export PATH=/usr/local/bin:$UBUNTU_HOME/.cargo/bin:$PATH
export CARGO_TARGET_DIR=$UBUNTU_HOME/work/camelid-targets/backend-95495a91
cd $UBUNTU_HOME/work/camelid-cron-95495a91-20260522T161257Z-main
node scripts/test-bench-llama3-same-host.mjs
