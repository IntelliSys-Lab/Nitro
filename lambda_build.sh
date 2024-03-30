#! /bin/bash

set -ex

# Vars
IMAGE_REPO="aws_ecr"
IMAGE_TAG="serverless_actor"

# Build Lambda image
cd aws_lambda
sudo docker build --platform linux/amd64 -t $IMAGE_REPO:$IMAGE_TAG .

cd ../
./lambda_local_test.sh
