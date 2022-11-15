.. _contribpr:

PR workflow
===========

The PR, short for Github Pull Request, is a way to merge code on Github, where the main Arbor-repo is hosted.

.. _contribpr-issue:

Issues
------

New features, bugfixes or other kinds of contributions ideally start their lives as an Issue (short for Github Issue)
on our `Issue tracker <https://github.com/arbor-sim/arbor/issues>`_. We distinguish two broad categories of
issues: feature requests to add new functionality to arbor and bugs pointing out mistakes, inaccuracies,
and plain faults in Arbor. Having a formal Github Issue before an implementation addressing a request or bug
(as a PR containing a code contribution or otherwise) gives others the chance to weigh in and help
find a solution that fits Arbor and its design, which makes it easier to integrate your contribution.
Especially for new features, this is a helpful process. Have a look at our
`blogpost on this subject <https://arbor-sim.org/how-to-file-an-issue/>`_ for some more rationale for
this process.

.. _contribpr-make:

Making a pull request
---------------------

If you want to contribute code that implements a solution or feature,
the workflow is as follows:

1. Fork the Arbor repository.
2. Create a branch off of master and implement your feature.
3. Make a pull request (PR) and refer to the issue(s) that the PR
   addresses. Some tips on how to write a good PR description:

   - You can use labels to categorize your PR. For very large PRs, it
     is likely the reviewer will have many questions and comments. It
     can be helpful to reach out on Gitter, Github Discussions or by email
     beforehand if you’ve big things planned for Arbor!
   - Commit logical units with clear commit messages.
   - Add a change summary to the PR message. If you remembered to commit
     logical units, this could simply be a bulleted list of all commit
     messages in the PR. If during the review process you make changes
     to the features in the PR, keep the PR description updated.
   - Use text like `fixes #123` in the description to refer to an issue.
   - Prefix your PR title with BUGFIX, BREAKING, FEATURE if it fixes a bug, introduces
     a breaking change or introduces a (major) new feature respectively.

4. We will accept contributions licensed with the same
   `BSD 3-Clause "New" or "Revised" License <https://github.com/arbor-sim/arbor/blob/master/LICENSE>`_,
   as the Arbor project.
   If not specified otherwise, we accept your contribution under this license.
   If this is a problem for you, please contact us at
   `contact@arbor-sim.org <mailto:contact@arbor-sim.org>`__.
5. A PR on Github is automatically built and tested by GitHub Actions CI pipelines.
   You can run these tests yourself by building them first
   (``make tests``) and then running them (``./bin/*unit*``).

   -  If you make changes effecting simulations on GPUs, you can post a reply to
      your PR with ``bors try``, which will run the test-suite with GPU and MPI
      enabled on `Piz Daint <https://www.cscs.ch/computers/piz-daint/>`_.
6. A member of the Arbor development team will review your contribution.
   If they approve, your PR will be merged! Normally this should happen
   within a few days.

Refer to the `Github
documentation <https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request>`__
for more explanation of the Git workflow described above.

.. _contribpr-collab:

Collaborating on a PR
---------------------

Especially for larger PRs, it may be a good idea to collaborate with others. There are various ways to do so,
for instance to have a shared branch somewhere in a fork of the Arbor repo. Since we're using Github, you can
also add commits to a PR opened by someone else. Since the correct procedure for that is 
`Github specific <https://docs.github.com/en/github/collaborating-with-pull-requests/working-with-forks/allowing-changes-to-a-pull-request-branch-created-from-a-fork>`_,
here the steps for how to do this (with just ``git``).

Alongside the instruction, an example situation will be given. Here, it is assumed that you've created your own
Github fork of the Arbor repo, and that you have this cloned to disk (and the ``origin`` remote points to your
fork of Arbor on Github). We'll setup a new remote called ``upstream``, pointing to the ``arbor-sim/arbor`` repo.
The example situation is that you want to commit to pull request number ``321``, opened by a 
Github user called ``github_user_007``, who named the branch they're trying to merge ``my_special_branch``,
in their fork of the Arbor repo called ``arbor-sim`` (which is the default when you fork on Github).

Situation as a table
~~~~~~~~~~~~~~~~~~~~

=============================== ========================= ======================
description                     variable                  example
remote for Arbor main repo      ``$REMOTE``               ``upstream``
PR you want to contribute to    ``$PR``                   ``321``
Github user that opened the PR  ``$PR_AUTHOR``            ``github_user_007``
branch name of the PR           ``$REMOTE_BRANCH_NAME``   ``my_special_branch``
repo name of the above branch   ``$REPO``                 ``arbor-sim``
=============================== ========================= ======================

Steps
~~~~~

0. Add the ``upstream`` remote like so:
   ``git remote add upstream https://github.com/arbor-sim/arbor``
1. ``git fetch $REMOTE pull/$PR/head:$REMOTE_BRANCH_NAME``

   Example: ``git fetch upstream pull/321/head:my_special_branch``
2. This should have made a local branch with the same name, tracking the PR-branch. Switch to the new local branch.

   Example: ``git switch my_special_branch``
3. Make commits to that local branch.
4. Push to the PR-branch with the following incantation:
   ``git push git@github.com:$PR_AUTHOR/$REPO $LOCAL_BRANCH_NAME:$REMOTE_BRANCH_NAME``

   Example: ``git push git@github.com:github_user_007/arbor-sim my_special_branch:my_special_branch``
5. The commit should now show up on the PR. When the PR is going to be merged, Github will add a
   "Co-authored by ..." line to the commit body. Leaving this line in place upon merging, will then list
   these contributors in Github UI.

.. _contribpr-review:

Reviewing a PR
--------------

Each pull request is reviewed according to these guidelines:

-  At least one core Arbor team member needs to mark your PR with a
   positive review.
-  GitHub Actions CI must produce a favourable result on your PR.
-  An Arbor team member will (squash) merge the PR with the PR change
   summary as commit message.
-  Consider using Gitpod to review larger PRs, see under checks on the Github PR page.

.. _contribpr-lint:

Pull requests will also be subject to a series of automated checks

- Python formatting will be checked using the `black <https://black.readthedocs.io/en/stable/index.html>`__ formatter
- Python files will be checked for common errors and code smells using `flake8 <https://flake8.pycqa.org/en/latest/>`__
- C++ code will be run against a suite of sanitizers under the `clang <https://clang.llvm.org/docs/index.html>`__ umbrella. The following checks are enabled
  - `undefined behavior <https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html>`__: under/overflow, null-deref, ...
  - `threads <https://clang.llvm.org/docs/ThreadSanitizer.html>`__: data races and other threading related issues
  - `memory <https://clang.llvm.org/docs/AddressSanitizer.html>`__: illegal accesses, use-after-free, double free, ...

.. _contribpr-merge:

Merging a PR
------------

-  Use PR comment as commit message and verify it covers the changes in
   the PR.
-  ONLY squash-and-merge (Github should not allow anything else
   anymore).
-  The creator of a pull request should not review or merge their own
   pull request.
-  A reviewer can merge if their own review is favourable and other
   criteria are met.
-  A reviewer can poke another Arbor core team member to do the merge.
