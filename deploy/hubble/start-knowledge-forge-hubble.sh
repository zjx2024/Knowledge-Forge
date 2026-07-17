#!/bin/sh
set -eu

ln -sfn /hubble/data/db.mv.db /hubble/db.mv.db
exec /hubble/bin/start-hubble.sh -f true
