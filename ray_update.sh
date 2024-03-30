#! /bin/bash

set -e

cd ~/ray/python && pip install -e . --user --verbose && cd ~/serverless-rl
