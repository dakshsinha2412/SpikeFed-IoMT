#!/usr/bin/env bash
# deploy_to_pi.sh — Syncs updated modules, configs, and datasets to Pi
set -e

PI_HOST="192.168.137.2"
PI_USER="pi"
REMOTE_DIR="/home/pi/spikefed-iomt"

echo "Deploying SpikeFed-IoMT Edge Client to ${PI_USER}@${PI_HOST}:${REMOTE_DIR}..."

ssh "${PI_USER}@${PI_HOST}" "mkdir -p ${REMOTE_DIR}/config ${REMOTE_DIR}/data"

scp client.py model.py dataset.py dataset_preprocessing.py dataset_registry.py dataset_loader.py \
    dataset_validation.py dataset_statistics.py federated_partition.py \
    verify_pi_environment.sh pi_requirements.txt "${PI_USER}@${PI_HOST}:${REMOTE_DIR}/"

scp config/pi_client_config.json "${PI_USER}@${PI_HOST}:${REMOTE_DIR}/config/"

# Selective dataset sync
scp -r data/chest_xray "${PI_USER}@${PI_HOST}:${REMOTE_DIR}/data/"
scp -r data/ultrasound "${PI_USER}@${PI_HOST}:${REMOTE_DIR}/data/"

echo "Deployment complete. Running verification script on Pi..."
ssh "${PI_USER}@${PI_HOST}" "cd ${REMOTE_DIR} && chmod +x verify_pi_environment.sh && ./verify_pi_environment.sh"
