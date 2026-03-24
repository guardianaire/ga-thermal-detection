#!/usr/bin/env bash
# Deploy the server-side thermal detection container to Azure.
#
# Prerequisites:
#   - Azure CLI authenticated (az login)
#   - ACR credentials (fetched automatically below)
#   - Client certs in ./certs/ (client4.pem, client4.key)
#   - Model weights in ./weights/thermal_custom_best.pt
#
# Usage:
#   ./deploy.sh          # Build and deploy
#   ./deploy.sh build    # Build image only
#   ./deploy.sh deploy   # Deploy only (image must exist)

set -euo pipefail

RESOURCE_GROUP="ga-streaming-rg"
ACR_NAME="gastreamingacr"
ACR_SERVER="${ACR_NAME}.azurecr.io"
IMAGE_NAME="ga-thermal-detect"
IMAGE_TAG="latest"
CONTAINER_NAME="ga-thermal-detect"

ACTION="${1:-all}"

echo "=== GA Thermal Detection Server Deploy ==="
echo "Resource Group: $RESOURCE_GROUP"
echo "ACR: $ACR_SERVER"
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo ""

# Verify required files exist
if [[ "$ACTION" == "all" || "$ACTION" == "build" ]]; then
    for f in weights/thermal_custom_best.pt certs/client4.pem certs/client4.key; do
        if [ ! -f "$f" ]; then
            echo "ERROR: Required file missing: $f"
            exit 1
        fi
    done
fi

# Build
if [[ "$ACTION" == "all" || "$ACTION" == "build" ]]; then
    echo "--- Building image in ACR ---"
    az acr build \
        --registry "$ACR_NAME" \
        --image "${IMAGE_NAME}:${IMAGE_TAG}" \
        --file Dockerfile.server \
        .
    echo ""
fi

# Deploy
if [[ "$ACTION" == "all" || "$ACTION" == "deploy" ]]; then
    echo "--- Fetching ACR credentials ---"
    ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)

    echo "--- Deploying container instance ---"

    # Delete existing container if present (update not supported for all fields)
    az container delete \
        --resource-group "$RESOURCE_GROUP" \
        --name "$CONTAINER_NAME" \
        --yes 2>/dev/null || true

    az container create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$CONTAINER_NAME" \
        --image "${ACR_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}" \
        --registry-login-server "$ACR_SERVER" \
        --registry-username "$ACR_NAME" \
        --registry-password "$ACR_PASSWORD" \
        --os-type Linux \
        --cpu 2 \
        --memory 4 \
        --restart-policy Always \
        -o table

    echo ""
    echo "--- Container status ---"
    az container show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$CONTAINER_NAME" \
        --query "{state:instanceView.state, ip:ipAddress.ip, restarts:containers[0].instanceView.restartCount}" \
        -o table

    echo ""
    echo "To view logs:"
    echo "  az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --follow"
fi
