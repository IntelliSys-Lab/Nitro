#! /bin/bash

set -ex

# Vars
IMAGE_REPO="aws_ecr"
IMAGE_TAG="serverless_actor"

# Build Lambda image
cd aws_lambda
sudo docker run -p 9000:8080 $IMAGE_REPO:$IMAGE_TAG

# Stop and remove all containers
sudo docker stop $(sudo docker ps -aq)
sudo docker rm $(sudo docker ps -aq)
sudo docker system prune -f

cd ../
