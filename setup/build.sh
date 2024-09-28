#!/bin/bash

echo 'Building GPU image with name ipg-jax'
docker build \
    --build-arg UID=$(id -u ${USER}) \
    --build-arg GID=1234 \
    -t ipg-jax \
    --build-arg REQS="$(cat ./requirements.txt | tr '\n' ' ')" \
    .
