#!/bin/bash
docker compose up -d
sleep 2

docker exec ga_cf_planner bash -c "source /opt/ros/rolling/setup.bash && ./build.sh"

sleep 1
docker compose down