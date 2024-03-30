#!/bin/bash

set -ex

cd logs

for file in *.csv; do
    mv "$file" "${file%.csv}_bkp.csv"
done

cd ../