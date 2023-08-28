#!/usr/bin/env bash
# Runs all C++ examples

set -Eeuo pipefail

# Function to print the script's synopsis
print_synopsis() {
    echo "Usage: $(basename "$0") [-d] [--with=example] [prefix]"
    echo "  -d              Toggle distributed mode (optional). Does not expect result"
    echo "                  files to be available to every process."
    echo "  --with=example  Select an example (optional). If omitted all examples will run."
    echo "  prefix          Prefix value (optional). Will toggle distributed mode if it"
    echo "                  starts with mpirun*."
}

# Default values
distributed=0
examples=()
prefix=""
PREFIX=" `pwd`/build/bin"
tag=dev-`git rev-parse --short HEAD`
out="results/$tag/cpp/"
ok=0

# List of all examples
all_examples=(
    "bench"
    "brunel"
    "gap_junctions"
    "generators"
    "lfp"
    "ring"
    "busyring"
    "single-cell"
    "probe-demo v"
    "plasticity"
    "ou"
    "voltage-clamp"
    "network_description"
    "remote"
)

# Mark examples not suitable for local execution
skip_local=(
    "remote"
)

# Lookup table for expected spike count
expected_outputs=(
    972
    6998
    "30"
    ""
    ""
    94
    35000
    ""
    ""
    ""
    ""
    ""
    205
    ""
)

# Function to execute an example
execute_example() {
    local example="${1}"
    local dir=`echo ${example} | tr ' ' '_'`
    local path="${out}${dir}"
    echo -n "   - ${example}: "

    # skip marked examples if we are in distributed mode
    if [[ $distributed == 0 ]]; then
        for ex in "${skip_local[@]}"; do
            if [[ $ex == $example ]]; then
                echo "skipped"
                return
            fi
        done
    fi

    # run the example and redirect its output
    mkdir -p ${path}
    ${PREFIX}/${example} > ${path}/stdout.txt 2> ${path}/stderr.txt

    # get the expected output if it exists and compare it to the actual output
    local expected=""
    for i in "${!all_examples[@]}"; do
        if [[ ${all_examples[$i]} == $example ]]; then
            expected=${expected_outputs[$i]}
            break
        fi
    done
    if [[ -n ${expected} ]]; then
        actual=$(grep -Eo '[0-9]+ spikes' ${path}/stdout.txt || echo "N/A")
        if [[ $distributed == 1 && "$actual" == "N/A" ]]; then
            echo "check skipped on remote rank"
        elif [ "$expected spikes" == "$actual" ]; then
            echo "OK"
        else
            echo "ERROR wrong number of spikes: $expected ./. $actual"
            ok=1
        fi
    else
        echo "OK (nothing to check)"
    fi
}

# Argument parsing
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        -d)
        distributed=1
        shift
        ;;
        --with=*)
        example="${key#*=}"
        if [[ " ${all_examples[@]} " =~ " $example " ]]; then
            examples+=("$example")
        else
            echo "Error: Invalid example '$example'"
            print_synopsis
            exit 1
        fi
        shift
        ;;
        *)
        if [[ $key == -* ]]; then
            echo "Error: Invalid argument '$key'"
            print_synopsis
            exit 1
        fi
        prefix="$key"
        shift
        ;;
    esac
done

# If --with=example was not used, add all entries from all_examples to examples
if [[ ${#examples[@]} -eq 0 ]]; then
    examples=("${all_examples[@]}")
fi

## Set distributed to true if prefix is not empty
#if [[ -n $prefix ]]; then
# Set distributed to true if prefix starts with mpirun
if [[ $prefix == mpirun* ]]; then
    distributed=1
fi

# Concatenate full prefix
PREFIX="${prefix}${PREFIX}"

# Execute the selected examples
for example in "${examples[@]}"; do
    execute_example "${example}"
done

exit $ok
