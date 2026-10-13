# Release Process

**Purpose**: Outline the general steps taken when cutting a new Arkouda release

_Our release process is continually evolving._
When a general consensus forms around cutting a new release a developer should follow these general steps.

## Step-by-step instructions
- Gain consensus from the core development team and confirm the current HEAD of the `main` branch builds and passes all tests on the continuous integration workflow (we use Github's built in Actions Workflows for this).
- Since pulling together the release notes tends to be the long pole in the tent and the name of the release is based on the date, start by drafting the release notes (see next section), ideally sorting changes into categories, and roughly by importance.  Get this reviewed by someone who's pretty familiar with the changes..
- Navigate to the `releases` page in the Arkouda project ([Direct Link](https://github.com/Bears-R-Us/arkouda/releases)).  You can navigate to it from the `Code` tab of the Arkouda project and locating the `Releases` section on the right and side of the page.
- Click the `Draft a new release` button which will bring you to a new page.
- From the `Tag: Select tag` drop-down, select `Create new tag` and in the text input box add the new version number.  This tag will be _auto-created_ on the final step when we _Publish_ the release.
  - As of this writing our versioning scheme is date based following the pattern `v<YYYY>.<MM>.<DD>`.  Example:  `v2022.01.31`.  Please note we zero-pad the month & day as necessary to ensure lexicographic sorting, and we are using a dot `.` as the delimiter.
  - Should it be necessary to tag more than one release on the same day (hey, mistakes happen it's ok), we use a dash and incrementing whole number, i.e. `v2022.01.31-1`, `v2022.01.31-2`, etc.
- Add a `Release Title` in the appropriate text box. We use the tag name.
- In the larger `Describe the release` text box you should add the release notes.
- When everything has been completed, you can optionally click the `Save draft` button at the bottom so someone else can review the release notes etc. Or go straight to the next step.
- Once everything is ready click the green `Publish release` button.
- Post a link and share the news with the community.
- Update the Spack Arkouda formula to support the new release
  - it lives here: https://github.com/spack/spack-packages/tree/develop/repos/spack_repo/builtin/packages/arkouda
  - and here's a sample PR that updates it: https://github.com/spack/spack-packages/pull/6700

## Generating release notes
We believe it is important to generate good release notes; highlighting significant changes, minor changes, and as a fail-safe using Github's `+ Auto-generate release notes` feature.  The following steps can be loosely followed to determine what is in the release itself OR you can `auto-generate` them and go through them to determine which change-sets are significant.  When possible add a link to the Issue number and/or PR number.

### Diff the git logs
We are going to assume you have named the Bears-R-Us remote `upstream` in the following git commands. If you named it something else, replace it as necessary.
- `git fetch upstream --tags` to make sure you have the previous tags from upstream.
- `git tag -l` to list the tags; you should see the most recent tag in this list.
- Use one of the following commands to get an idea of the commit history by comparing the last tag with the current state of the `upstream` remote.  We use git's ellipse operator `..`
  - `git log <prev-tag>..upstream/main` ; for example `git log v2022.01.20..upstream/main`
  - Use the `--oneline` option to make the output more concise: `git log --oneline v2022.01.20..upstream/main`
- Alternatively you can use the graphical `gitk` utility to view the history.
- We have been pushing developers to include the Issue number in their commits which makes it easy to see which issues are associated with the new release. If this is not listed, you should be able to retrieve the pull request number associated with the merged commit(s). You will need to use the Github web interface to find the associated PR and check its description to find any associated issue numbers.  **You should strive to link ALL related issues and/or PRs in the release notes**.  Github generally automates the hyperlink creation using defined patterns like `#1234`.
- Once you have the Issue & PR numbers, you will need to determine what is considered _Major_ and _Minor_ with respect to new functionality. A reasonable guideline for considering changes
  - Major
    - New features (user facing)
    - Substantial, internal design changes for developers
    - Substantial performance improvement
    - Deprecation of an old feature
    - Bug fixes
  - Minor
    - Smaller performance improvements
    - Internal changes with no real user-facing impact
    - (Basically anything which doesn't fit in the Major category)
- Auto-generated changes: As a fail-safe we like to add a third section to the Release notes using Github's `+ Auto-generate release notes` button on the upper right hand side of the release-notes description text field.

When in doubt look at previous releases to get an idea of what our release notes look like.