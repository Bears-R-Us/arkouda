#!/bin/bash

#  Intended to be called from the arkouda directory
#  Example usage:
#  bash docker/build_arkouda_container.sh -i ubuntu-with-arkouda-deps -v 1.0.0 -r ajpotts


set -e  # Exit immediately on error

# Function to display usage
usage() {
    echo "Usage: $0 -v VERSION -i IMAGE_NAME -r REPO_NAME [-p]"
    echo ""
    echo "Options:"
    echo "  -v VERSION       Version tag for the Docker image (e.g., 1.0.0)"
    echo "  -i IMAGE_NAME    Name of the Docker image (e.g., ubuntu-with-arkouda-deps)"
    echo "  -r REPO_NAME     Docker repository name (e.g., ajpotts)"
    echo "  -p               Push the image to the repository"
    exit 1
}

# Initialize variables
PUSH_IMAGE=false

# Parse command line options
while getopts ":v:i:r:p" opt; do
  case $opt in
    v) VERSION="$OPTARG" ;;
    i) IMAGE_NAME="$OPTARG" ;;
    r) REPO_NAME="$OPTARG" ;;
    p) PUSH_IMAGE=true ;;
    \?) echo "Invalid option -$OPTARG" >&2; usage ;;
    :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
  esac
done

# Check for required arguments
if [ -z "$VERSION" ] || [ -z "$IMAGE_NAME" ] || [ -z "$REPO_NAME" ]; then
    usage
fi

echo "This script is intended to be run from the arkouda project directory."

docker build -t "$IMAGE_NAME:$VERSION" "docker/$IMAGE_NAME/$VERSION/"
docker tag "$IMAGE_NAME:$VERSION" "$REPO_NAME/$IMAGE_NAME:$VERSION"

if [ "$PUSH_IMAGE" = true ]; then
    echo "Pushing image to repository..."
    docker push "$REPO_NAME/$IMAGE_NAME:$VERSION"
else
    echo "Push flag not set. Skipping image push."
fi
