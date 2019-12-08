#!/bin/bash


if [[ ! -z "$1" ]]; then
    python3 -m tensorboard.main --logdir=$1 --host localhost

else
    echo "Usages:"
    echo "    $0 <logdir>"
fi
