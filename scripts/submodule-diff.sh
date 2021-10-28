#!/usr/bin/env bash

git submodule foreach 'git describe HEAD --tags' > pre.log
git submodule foreach 'git fetch'
git submodule foreach 'git describe `git log --branches -1 --pretty=format:"%H"` --tags --abbrev=0' > post.log
if diff pre.log post.log ; then
    echo "No submodule updates found"
else
    diff pre.log post.log -U 10000 > diff.log
fi
rm pre.log && rm post.log
