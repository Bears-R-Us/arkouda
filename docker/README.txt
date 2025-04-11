# Docker Build and Push Script

This script builds and optionally pushes a Docker image for the Arkouda project.

## Prerequisites

- Docker installed and running
- Logged in to your Docker registry (e.g., via `docker login`)
- This script should be run from the root of the Arkouda project directory

## Usage

```bash
bash docker/build_arkouda_container.sh -v VERSION -i IMAGE_NAME -r REPO_NAME [-p]
```
