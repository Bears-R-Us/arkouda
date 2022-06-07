# Contributing to Arkouda

Arkouda is an open source project and we love to see new contributors! We welcome contributions via issues and pull requests

<a id="toc"></a>
## Table of Contents

1. [Adding Issues](#issues)
   - [Bug Reports](#bug-reports)
   - [Feature Requests](#feature-requests)
2. [Developing Arkouda](#development)
   - [Coding Conventions and Linting](#code-quality)
   - [Testing](#testing)
   - [Writing Pull Requests](#writing-prs)
3. [Core Development Team Only](#core-dev)
   - [Merging Pull Requests](#merging-prs)
   - [Release Process](#release)

<a id="issues"></a>
## Adding Issues <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
Most issues fall under the broad categories of bug reports or feature requests.
If your issue doesn't fit either of these, please add it anyway and provide as much detail as possible

Using github markdown (especially [code blocks](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-and-highlighting-code-blocks))
is very appreciated

<a id="bug-reports"></a>
### Bug Reports <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
- Summary of Problem
  - What behavior did you observe when encountering the bug?
  - What behavior did you expect to observe?
  - Is this a blocking issue with no known work-arounds?
- Steps to Reproduce
  - Please provide code that will reproduce the problem if possible.
Providing simplified programs demonstrating the problem will be appreciated
- Configuration Information
  - What's the output of `ak.get_config()['arkoudaVersion']`

<a id="feature-requests"></a>
### Feature Requests <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
Be as specific as possible and provide examples when appropriate. If the requested feature is based on another library 
(i.e. numpy, pandas, scipy), add a link to their documentation on it

<a id="development"></a>
## Developing Arkouda <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

If you don't have anything in mind, check out our [outstanding issues](https://github.com/Bears-R-Us/arkouda/issues) 
(it's likely a good idea to filter on the label `good first issue`).
If you find an issue you'd like to work, be sure to comment indicating that! So we avoid someone accidentally duplicating your work

If you already have an idea, be sure to [first add an issue](#issues) detailing the feature or bug fix and
indicate that you want to work it! This way we can discuss it before you go through the trouble of writing the code

We use a [Git Forking Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow)
and recommend developers use a simple [Git Feature Branch Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow)
on their own fork

<a id="code-quality"></a>
### Coding Conventions and Linting <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

#### Python3
We follow the coding standards laid out in [PEP8](https://peps.python.org/pep-0008/). 
Our Continuous Integration workflow has a linter (`flake8`) to verify all our python code meets these requirements

We use `isort`, `black`, and `flake8` (typically in that order) to ensure our code is consistent.
We have decided on 105 for our line length instead of the more restrictive default of 88 provided by `black`

```bash
$ isort arkouda/example_feature.py
Fixing arkouda/example_feature.py

$ black --line-length 105 arkouda/example_feature.py
reformatted arkouda/example_feature.py

All done!
1 file reformatted.

$ flake8 arkouda/example_feature.py
```
For users of pycharm there is nice [interoperability](https://black.readthedocs.io/en/stable/integrations/editors.html#pycharm-intellij-idea) with these tools

We use numpy style doc strings. It's a good idea to write code that looks similar to the surrounding functions

#### Chapel
Our chapel code is space only. The CI has a linter which will fail if tabs are present
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
<a id="testing"></a>
### Testing <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

If you’re fixing a bug, add a test. Run it first to confirm it fails, then fix the bug, run it again to confirm it’s really fixed

If adding a new feature, add a test to make sure it behaves properly

See our wiki for more info on how to run our tests and create your own:
https://github.com/Bears-R-Us/arkouda/wiki/Unit-Testing

#### Running python tests
```terminal
make test
```
#### Running chapel tests
```terminal
python3 util/test/parallel_start_test.py -d test
```

<a id="writing-prs"></a>
### Writing Pull Requests <sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
Every pull request should have at least one associated issue (if there's not one, create one!)
 
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

<a id="core-dev"></a>
## Core Development Team Only<sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

<a id="merging-prs"></a>
### Merging Pull Requests<sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>
- Only members of the core dev team with quite a bit of experience writing and reviewing PRs should merge pull requests. If you're unsure, ask first
- Only merge PRs with 2 or more concurrent approvals from members of the core dev team with limited exceptions
- Only merge PRs after the CI has passed and if there are no conflicts. Ideally rebase or merge with master first
- If you wrote it, don't merge it. It's best practice for someone else to decide it's ready to be merged

<a id="release"></a>
### Release Process<sup><sup><sub><a href="#toc">toc</a></sub></sup></sup>

New versions should only be released after the core dev team has reached a consensus.
Instructions on our release process:

https://github.com/Bears-R-Us/arkouda/wiki/Release-Process