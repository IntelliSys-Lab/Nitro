#!/bin/bash

set -ex

REDIS_PORT=6380
REDIS_PASSWORD="Nitro"

redis-cli -p $REDIS_PORT -a $REDIS_PASSWORD monitor
