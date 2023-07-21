#!/usr/bin/env bash
# Runs tests of executables

set -Eeuo pipefail

echo "=== Executing modcc test ======================================"
modcc python/example/cat/dummy.mod
test -f "dummy.hpp"
if [[ ! -z "$GITHUB_ACTIONS" ]]; then
  echo "### modcc: OK." >> $GITHUB_STEP_SUMMARY
fi

# echo "=== Executing a-b-c test ======================================"
# arbor-build-catalogue -v cat python/example/cat
# ./scripts/test-catalogue.py ./cat-catalogue.so
# if [[ ! -z "$GITHUB_ACTIONS" ]]; then
#   echo "### a-b-c: OK." >> $GITHUB_STEP_SUMMARY
# fi
