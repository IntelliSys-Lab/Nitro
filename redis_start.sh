#!/bin/bash

set -ex

REDIS_PORT=6379
REDIS_PASSWORD="Nitro"

sudo redis-server /etc/redis/redis.conf
redis-cli -p $REDIS_PORT -a $REDIS_PASSWORD ping 
