#! /bin/bash

# Start a ray node
/usr/local/bin/ray start --address=${RAY_HEAD_IP}:${RAY_HEAD_PORT} --disable-usage-stats

# Keep container running
tail -f /dev/null
