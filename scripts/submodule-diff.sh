#!/usr/bin/env bash

git submodule foreach 'git describe HEAD --tags' > current_state_of_git_submodules_in_arbor_repo.log | tee current_state_of_git_submodules_in_arbor_repo.log
git submodule foreach 'git fetch'
git submodule foreach 'git describe `git log --branches -1 --pretty=format:"%H"` --tags --abbrev=0' > upstream_state_of_git_submodules.log | tee upstream_state_of_git_submodules.log
if diff current_state_of_git_submodules_in_arbor_repo.log upstream_state_of_git_submodules.log ; then
    echo "No submodule updates found"
else
    diff current_state_of_git_submodules_in_arbor_repo.log upstream_state_of_git_submodules.log -U 10000 > diff.log
fi
rm current_state_of_git_submodules_in_arbor_repo.log
rm upstream_state_of_git_submodules.log
