#!/usr/bin/env bash
# Check any git submodule remotes for updates, and print difference with current Arbor repo state to `diff.log`

git submodule foreach 'git describe HEAD --tags' | tee current_state_of_git_submodules_in_arbor_repo.log
git submodule foreach 'git fetch'
git submodule foreach 'git describe `git log --branches -1 --pretty=format:"%H"` --tags --abbrev=0' | tee upstream_state_of_git_submodules.log
if diff current_state_of_git_submodules_in_arbor_repo.log upstream_state_of_git_submodules.log ; then
    echo "No submodule updates found"
else
    diff current_state_of_git_submodules_in_arbor_repo.log upstream_state_of_git_submodules.log -U 10000 > diff.log
fi
rm current_state_of_git_submodules_in_arbor_repo.log
rm upstream_state_of_git_submodules.log
