# Contributing to Arbor

First off, thank you for considering contributing to Arbor! It's people like you that
make Arbor such a great tool. Feel welcome and read the following sections in order to
know how to ask questions and how to work on something.

## Types of contributions

There are many ways to contribute: writing tutorials or blog posts, improving the
documentation, submitting bug reports and feature requests or writing code which can be
incorporated into Arbor itself.

If you come across a bug in Arbor, please [file an issue on the GitHub issue tracker](https://github.com/arbor-sim/arbor/issues/new).
You can attach files to issues. Data and bits of code that help to reproduce the issue
are very welcome; they'll make it easier for Arbor developers to debug and fix your issue.

The source for [the documentation](https://arbor.readthedocs.io) is found in the `/doc` subdirectory,
C++ examples in `/example` and Python examples in `/python/example`. You can add yours in the same way
you would contribute code, please see the "Github workflow" section.

Please don't use the issue tracker for support questions. You can use the [Arbor Discussions](https://github.com/arbor-sim/arbor/discussions)
page for this.

## Github workflow

Arbor uses Git for version control and Github to host its code repository and nearly
all of its infrastructure.

* If you're not familiar with Git or Github, please have at look at
[this introduction](https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/set-up-git).
* Make sure you have a [GitHub account](https://github.com/signup/free).

### Start a discussion

You can use the [Arbor Discussions](https://github.com/arbor-sim/arbor/discussions) to help other users of Arbor,
or get help with you own simulations. Also, we're eager to discover what you're using Arbor for, so don't hesitate
to share your Arbor simulations or publications!

### Filing an issue

If you have found a bug or problem in Arbor, or want to request a feature, you can use our
[issue tracker](https://github.com/arbor-sim/arbor/issues). If you issue is not yet filed in the issue tracker,
please do so and describe the problem, bug or feature as best you can. You can add supporting data, code or documents
to help make your point.

### Making a pull request

If you want to contribute code that implements a solution or feature, the workflow is as follows:

1. Fork the Arbor repository
2. Create a branch off of master and implement your feature
3. Make a pull request (PR) and refer to the issue(s) that the PR addresses.
4. An administrative matter: Arbor requests [an explicit copyright assignment](https://en.wikipedia.org/wiki/Copyright_transfer_agreement)
on your contribution. This grants us the right to defend copyright on your contributions on your behalf,
and allows us to change the license on your contribution to another open source license should that become
necessary in the future. You can upload a filled out copy of the agreement as a file attached to the PR, or
if you prefer not to disclose private information, you can send it to <mailto:arbor-sim@fz-juelich.de>.
    * [Please find the Copyright Transfer Agreement here](https://github.com/arbor-sim/arbor-materials/tree/master/copyright-transfer-agreement).
5. A PR on Github is automatically tested by our CI bot called Travis. You can also run these tests yourself
by building them first (`make tests`) and then running them (`./bin/*unit*`).
6. A member of the Arbor development team will review your contribution. If they approve,
your PR will be merged! Normally this should happen within a few days.

Refer to the [Github documentation](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request)
for more explanation of the Git workflow described above.

### Reviewing a PR

Each pull request is reviewed according to these guidelines:

* At least one core Arbor team member needs to mark your PR with a positive review. Some tips to
help you get a positive review:
    * You can use labels to categorize your PR. For very large PRs, it is likely the reviewer will have
many questions and comments. It can be helpful to reach out on our Slack, Discussions or by email
beforehand if you've big things planned for Arbor!
    * Commit logical units with clear commit messages.
    * Add change summary to the PR message. If you remembered to commit logical units, this could simply be a bulleted list of all commit messages in the PR.
    * Make sure your code conforms to our [coding guidelines](https://github.com/arbor-sim/arbor/wiki/Coding-Style-Guidelines)
    * If you add functionality, please update the documentation accordingly.
    * If you add functionality, add tests if applicable. This helps make sure Arbor is stable and
    functionality does what it's supposed to do.
    * If you make changes to GPU simulations, you can post a reply to your PR with `bors try`. This will run our GPU test-suite.
    * If you add/change the public C++ API, provide Python wrappings.
    * Make sure Arbor compiles and has no new warnings.
* Travis CI must produce a favourable result on your PR.
* The creator of a pull request should not review or merge their own pull request.
* An Arbor team member will (squash) merge the PR with the PR change summery as commit message.

## Get in touch

You can reach out in the following ways:

* [Discussions](https://github.com/arbor-sim/arbor/discussions). Any questions or remarks regarding using Arbor
for your research are welcome.
* [Slack](https://mcnest.slack.com). If you're interested in developing Arbor itself, you can visit our Slack.
* [Email](mailto:arbor-sim@fz-juelich.de).
