#!/usr/bin/env bash
# Runs tests of executables

set -Eeuo pipefail

echo "=== Executing modcc test ======================================"
modcc python/example/cat/dummy.mod
test -f "dummy.hpp"

echo "=== Executing a-b-c test ======================================"
arbor-build-catalogue -v cat python/example/cat
./scripts/test-catalogue.py ./cat-catalogue.so
