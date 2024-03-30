#! /bin/bash

set -ex

# AWS IAM
aws_access_key_id="YOUR_AWS_ACCESS_KEY_ID"
aws_secret_access_key="YOUR_SECRET_ACCESS_KEY"
default_region="YOUR_REGION"

# Set the IAM
aws configure set aws_access_key_id $aws_access_key_id
aws configure set aws_secret_access_key $aws_secret_access_key
aws configure set default.region $default_region
