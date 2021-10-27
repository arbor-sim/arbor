#!/usr/bin/env bash

git submodule foreach 'git describe HEAD --tags' > pre.log
git submodule update --remote
git submodule foreach 'git checkout `git describe --abbrev=0 --tags`'
git submodule foreach 'git describe HEAD --tags' > post.log
if diff pre.log post.log ; then
    echo "No submodule updates found"
else
    diff pre.log post.log -U 10000 > diff.log
fi
echo "Last retval must be zero for GA to pass..."
