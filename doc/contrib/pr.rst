.. _contribpr:

PR workflow
===========

.. _contribpr-make:

Making a pull request
---------------------

If you want to contribute code that implements a solution or feature,
the workflow is as follows:

1. Fork the Arbor repository.
2. Create a branch off of master and implement your feature.
3. Make a pull request (PR) and refer to the issue(s) that the PR
   addresses. Some tips on how to write a good PR description:

   -  You can use labels to categorize your PR. For very large PRs, it
      is likely the reviewer will have many questions and comments. It
      can be helpful to reach out on our Slack, Discussions or by email
      beforehand if you’ve big things planned for Arbor!
   -  Commit logical units with clear commit messages.
   -  Add a change summary to the PR message. If you remembered to commit
      logical units, this could simply be a bulleted list of all commit
      messages in the PR. If during the review process you make changes
      to the features in the PR, keep the PR description updated.
   -  Use text like `fixes #123` in the description to refer to an issue.
4. An administrative matter: Arbor requests `an explicit copyright
   assignment <https://en.wikipedia.org/wiki/Copyright_transfer_agreement>`__
   on your contribution. This grants us the right to defend copyright on
   your contributions on your behalf, and allows us to change the
   license on your contribution to another open source license should
   that become necessary in the future. You can upload a filled out copy
   of the agreement as a file attached to the PR, or if you prefer not
   to disclose private information, you can send it to
   `arbor-sim@fz-juelich.de <mailto:arbor-sim@fz-juelich.de>`__.

   -  `Please find the Copyright Transfer Agreement
      here <https://github.com/arbor-sim/arbor-materials/tree/master/copyright-transfer-agreement>`__.

5. A PR on Github is automatically tested by our CI bot called Travis.
   You can also run these tests yourself by building them first
   (``make tests``) and then running them (``./bin/*unit*``).
6. A member of the Arbor development team will review your contribution.
   If they approve, your PR will be merged! Normally this should happen
   within a few days.

Refer to the `Github
documentation <https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request>`__
for more explanation of the Git workflow described above.

.. _contribpr-review:

Reviewing a PR
--------------

Each pull request is reviewed according to these guidelines:

-  At least one core Arbor team member needs to mark your PR with a
   positive review.
-  Travis CI must produce a favourable result on your PR.
-  An Arbor team member will (squash) merge the PR with the PR change
   summery as commit message.
-  Consider using Gitpod to review larger PRs (see under checks).

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
