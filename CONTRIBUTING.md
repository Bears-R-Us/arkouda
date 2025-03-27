# Contributing

Arkouda is an open source project and we love to see new contributors!
We welcome contributions via issues and pull requests.

## Adding Issues

Most issues fall under the broad categories of bug reports or feature requests.
If your issue doesn't fit either of these, please add it anyway and provide as much detail as possible.

It is always a good idea to review the current issue list and make sure your issue is not already present.

Using github markdown (especially [code blocks](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-and-highlighting-code-blocks))
is very appreciated.

### Bug Reports

When reporting a bug, please be sure to include the following information:

- Summary of Problem
  - What behavior did you observe when encountering the bug?
  - What behavior did you expect to observe?
  - Is this a blocking issue with no known work-arounds?
- Steps to Reproduce
  - Please provide code that will reproduce the problem if possible.
Providing simplified programs demonstrating the problem will be appreciated.
- Configuration Information
  - What's the output of `ak.get_config()`? This includes information like the `ArkoudaVersion` and the version of Chapel the server was built with.

### Feature Requests

Be as specific as possible and provide examples when appropriate. If the requested feature is based on another library 
(i.e. numpy, pandas, scipy), please provide a link to their supporting documentation for reference.

## Developing Arkouda

If you don't have anything in mind, check out our [outstanding issues](https://github.com/Bears-R-Us/arkouda/issues) 
(it's likely a good idea to filter on the label `good first issue`).

If you already have an idea for a new feature or have identified a bug, [add an issue](#issues) before you start working on it.

Once you find or create an issue you intend to work, please leave a comment in the issue indicating that.
Be sure to mention `@Bears-R-Us/arkouda-core-dev` for our awareness.
We will then assign the issue to you to avoid anyone duplicating your work.

Need assistance or want to discuss design with someone on the Arkouda team?
Add a comment tagging `@Bears-R-Us/arkouda-core-dev` to the issue and a developer will reach out!


We use a [Git Forking Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow)
and recommend developers use a simple [Git Feature Branch Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow)
on their own fork.

### Coding Conventions and Linting

#### Python3

We follow the coding standards laid out in [PEP8](https://peps.python.org/pep-0008/). 
Our Continuous Integration workflow has a linter (`flake8`) to verify all our python code meets these requirements.

We use `isort`, `black`, and `flake8` (typically in that order) to ensure our code is consistent.
We utilize a line length of 105. When running `black`, be sure to to use the `--line-length 105` parameter.
Please use `make ruff-format` and `make isort` to ensure consistency across contributors.

```bash
$ make isort
isort --gitignore --float-to-top arkouda
Skipped 34 files
isort --gitignore --float-to-top tests
Skipped 1 files

$ black --line-length 105 arkouda/example_feature.py
All done! ✨ 🍰 ✨
27 files reformatted, 67 files left unchanged.

$ flake8 arkouda/example_feature.py
```
For users of pycharm there is nice [interoperability](https://black.readthedocs.io/en/stable/integrations/editors.html#pycharm-intellij-idea) with these tools.

We use numpy style doc strings. It's a good idea to write code that looks similar to the surrounding functions.

#### Chapel

Our Chapel code linting is checking spaces only. The linter in the CI will fail if tabs are present.

* `lowerCamelCase` is used for variable names and procedures

```chapel
var aX: [{0..#s}] real;
proc printIt(x) {
     writeln(x);
}
```

 * `UpperCamelCase` is used for class names

```chapel
class Foo: FooParent {}
```

### Testing

If you’re fixing a bug, add a test. Run it first to confirm it fails, then fix the bug, run it again to confirm it’s really fixed.

If adding a new feature, add a test to make sure it behaves properly.

Things to note:
- If you make a new test, be sure to include `test_` at the beginning. Otherwise `pytest` will not run it.
- If you make a new file of tests, be sure to include the file in `pytest.ini`, so it will be run during a `make test`.

See our wiki for more info on how to run our tests and create your own:
https://github.com/Bears-R-Us/arkouda/wiki/Unit-Testing

#### Running python tests

```terminal
# Run all tests in pytest.ini
make test

# Run all tests in the CategoricalTest class (-v will print out the test name)
python3 -m pytest tests/categorical_test.py::CategoricalTest -v

# Run a single test from CategoricalTest named foo_test
python3 -m pytest tests/categorical_test.py::CategoricalTest::foo_test
```

#### Running chapel tests

```terminal
python3 server_util/test/parallel_start_test.py -d test
```

### Writing Pull Requests

Before posting a pull request, be sure to test locally to catch common CI failures early.
This usually includes running:
- `make test`
- `make mypy`
- `flake8 arkouda`

Every pull request should have at least one associated issue (if there's not one, create one!).

The issue number(s) should be listed in the title and the body using [closing keywords](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue).
Using github markdown (especially [code blocks](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-and-highlighting-code-blocks))
is very appreciated. When in doubt, take a look at some closed pull requests.

````
Closes #99999: Example Feature Request

This PR (Closes #99999):
- Implements example feature request
- Adds testing for example feature request

Example:
```python
>>> ak.example_feature(args)
"super cool functionality!"
```

Note:
- It will be helpful if reviewers keep THIS in mind
````

### Reviewing Pull Requests

For the most part, only the core dev team or those assigned should review PRs.

- As the person who left the PR feedback, you should be the one to resolve the conversation once you decide the author has addressed the issue.
- Try to resolve all of your feedback when you feel the PR is ready to be merged.
  - If necessary, add an issue to track any feedback which is outside the scope of the PR and link the PR comment in the issue.

## Core Development Team Only

### Merging Pull Requests

- Only members of the core dev team with quite a bit of experience writing and reviewing PRs should merge pull requests. If you're unsure, ask first.
- Only merge PRs with 2 or more concurrent approvals from members of the core dev team with limited exceptions.
- Only merge PRs after the CI has passed and if there are no conflicts. Ideally rebase or merge with master first.
- We prefer to have all feedback resolved before a PR is merged.
- If you wrote it, don't merge it. It's best practice for someone else to decide it's ready to be merged.
- To keep the commit history simple and allow for easy manipulation of PRs, we use the `Squash-and-Merge` functionality of GitHub's web interface.
- Be sure to include `@pierce314159` and `@Ethan-DeBandi99` as a reviewer

### Release Process

New versions should only be released after the core dev team has reached a consensus.
Follow our [release process](developer/RELEASE_PROCESS.md)
