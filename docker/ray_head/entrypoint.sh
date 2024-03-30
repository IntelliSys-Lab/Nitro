#! /bin/bash

# Start a ray head
/usr/local/bin/ray start --head --port=${RAY_HEAD_PORT} --disable-usage-stats

# Keep container running
tail -f /dev/null
