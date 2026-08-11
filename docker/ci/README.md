# docker CI images

Each folder in this directory contains a Dockerfile for building a container used in CI.

## Updating the images

To update the CI images, all you need to do is edit the relevant Dockerfile(s) and create a PR. The CI images are built automatically by `.github/workflows/build-CI-container-jobs.yml`. If the Dockerfiles are changed, the images will be rebuilt on a PR, and then again when the PR is merged to main. When the PR is merged, the updated images will be pushed to GitHub Container Registry (GHCR) under the Bears-R-Us organization.

## Making CI changes that require image updates

When making changes to the CI that require updates to the images, please follow these steps:

1. Edit the relevant Dockerfile(s) in this directory.
2. Open a PR with the changes to the Dockerfile(s).
3. Once the PR is open, the CI images will be rebuilt automatically. You can check the progress of the builds in the "Actions" tab of the repository.
4. Once the PR is approved and merged, the updated images will be pushed to GHCR automatically.
5. Make any additional changes to the CI workflows as needed, and open a new PR for those changes.

## Adding a new image

To add a new CI image, follow these steps:

1. Create a new directory in this `docker/` directory for the new image.
2. Add a Dockerfile to the new directory that defines the image.
3. Update `.github/workflows/build-CI-container-jobs.yml` to include jobs for building and pushing the new image. You can copy and modify the existing jobs for the other images as needed.
4. Open a PR with the new directory, Dockerfile, and workflow changes. The new image will be built and pushed automatically when the PR is merged.
5. Make any additional changes to the CI workflows as needed, and open a new PR for those changes.
