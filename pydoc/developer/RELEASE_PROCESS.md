# Release Process

**Purpose**: Outline the general steps taken when cutting a new Arkouda release

_Our release process is continually evolving._
When a general consensus forms around cutting a new release a developer should follow these general steps.

## Step-by-step instructions
- Gain consensus from the core development team and confirm the current HEAD of the `master`|`main` branch builds and passes all tests on the continuous integration workflow (we use Github's built in Actions Workflows for this).
- Navigate to the `releases` page in the Arkouda project ([Direct Link](https://github.com/Bears-R-Us/arkouda/releases)).  You can navigate to it from the `Code` tab of the Arkouda project and locating the `Releases` section on the right and side of the page.
- Click the `Draft a new release` button which will bring you to a new page.
- Select the `Choose a tag` drop down button and in the text input box add the new version number.  This tag will be _auto-created_ on the final step when we _Publish_ the release.
  - As of this writing our versioning scheme is date based following the pattern `v<YYYY>.<MM>.<DD>`.  Example:  `v2022.01.31`.  Please note we zero-pad the month & day as necessary to ensure lexicographic sorting, and we are using a dot `.` as the delimiter.
  - Should it be necessary to tag more than one release on the same day (hey, mistakes happen it's ok), we use a dash and incrementing whole number, i.e. `v2022.01.31-1`, `v2022.01.31-2`, etc.
- After inputting the new tag string, you can click the button underneath the text input box `+ Create new tag:<your tag> on publish`. You will see an informational note appear under the tag section with the text: _"Excellent! This tag will be created from the target when you publish this release."_
- Add a `Release Title` in the appropriate text box. We use the tag name.
- In the larger `Describe the release` text box you should add the release notes.  Ideally they should be split into the following. NOTE: We will go into more detail on how to generate the release notes in the next section.
  - Major changes
  - Minor changes
  - Auto-generated release notes
- When everything has been completed, you can optionally click the `Save draft` button at the bottom so someone else can review the release notes etc. Or go straight to the next step.
- Once everything is ready click the green `Publish release` button.
- Post a link and share the news with the community.

## Generating release notes
We believe it is important to generate good release notes; highlighting significant changes, minor changes, and as a fail-safe using Github's `+ Auto-generate release notes` feature.  The following steps can be loosely followed to determine what is in the release itself OR you can `auto-generate` them and go through them to determine which change-sets are significant.  When possible add a link to the Issue number and/or PR number.

### Diff the git logs
We are going to assume you have named the Bears-R-Us remote `upstream` in the following git commands. If you named it something else, replace it as necessary.
- `git fetch upstream --tags` to make sure you have the previous tab from upstream.
- `git tag -l` to list the tags; you should see the most recent tab in this list.
- Use one of the following commands to get an idea of the commit history by comparing the last tag with the current state of the `upstream` remote.  We use git's ellipse operator `..`
  - `git log <prev-tag>..upstream/master` ; for example `git log v2022.01.20..upstream/master`
  - Use the `--online` option to make the output more concise: `git log --oneline v2022.01.20..upstream/master`
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