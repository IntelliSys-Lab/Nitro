#! /bin/bash

set -ex

# Vars
AWS_REGION="YOUR_REGION"
AWS_ECR_URL="YOUR_ECR_URL"
AWS_LAMBDA_ROLE="YOUR_LAMBDA_ROLE"
AWS_LAMBDA_MEM_SIZE=1024
AWS_LAMBDA_TIMEOUT=60
IMAGE_REPO="aws_ecr"
IMAGE_TAG="serverless_actor"

# Connect to AWS ECR
aws ecr get-login-password --region $AWS_REGION | sudo docker login --username AWS --password-stdin $AWS_ECR_URL

# Push the image to AWS ECR
sudo docker tag  $IMAGE_REPO:$IMAGE_TAG $AWS_ECR_URL:$IMAGE_TAG
sudo docker push $AWS_ECR_URL:$IMAGE_TAG

# Update AWS Lambda function
aws lambda update-function-code \
  --function-name $IMAGE_TAG \
  --image-uri $AWS_ECR_URL:$IMAGE_TAG

cd ../
