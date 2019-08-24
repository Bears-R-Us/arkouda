# Git Utils

Git scripts to assist in managing any Git repository.

To use, simply run a script inside any Git repository. Optionally, you can
install the scripts into `git` by adding this directory to your `PATH`
environment variable in order to call a script via e.g., `git date`.


## git-date

Arguments: `[Branch (HEAD)]`

Output a time point for a commit reference.

## git-hash-from-date

Arguments: `Date [Branch (origin/master)] [Options]`

Output a commit's hash for a time point.

- `Date` must be any format that Git accepts via `--date`. See `git help show`,
  `PRETTY FORMATS` for valid committer's-date formats.

- `Branch` can be any valid commit reference.

- `Options` are optional flags accepted by `git rev-list`.
