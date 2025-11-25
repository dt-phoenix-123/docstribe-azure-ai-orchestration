#!/bin/bash
set -e

# Get the script's directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../" && pwd)"

echo "Script directory: $SCRIPT_DIR"
echo "Project root: $PROJECT_ROOT"

# Change to project root
cd "$PROJECT_ROOT"

# Check if client and environment arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <client> <environment> [repo-name]"
    echo "Example: $0 max production"
    echo "Example: $0 max dev ai-orchestration-custom-repo"
    echo ""
    echo "Available clients and environments:"
    find .envs/clients -name "*.env" 2>/dev/null | sed 's|.envs/clients/||g' | sed 's|.env||g' | sed 's|/| |g' || echo "No client environments found"
    exit 1
fi

CLIENT="$1"
ENVIRONMENT="$2"
ENV_FILE=".envs/clients/${CLIENT}/${ENVIRONMENT}.env"

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Environment file $ENV_FILE not found"
    echo "Please create the environment file with necessary configurations."
    echo "You can use .envs/ai-orchestration.env as a template."
    exit 1
fi

# Load environment variables
export $(cat "$ENV_FILE" | grep -v '^#' | grep -v '^$' | xargs)

# Allow repo name override via command line argument (3rd parameter)
if [ ! -z "$3" ]; then
    export AI_ORCHESTRATION_REPO_NAME="$3"
fi

# Validate required environment variables
if [ -z "$project_id" ]; then
    echo "Error: project_id not found in $ENV_FILE"
    exit 1
fi

if [ -z "$google_key_base64" ]; then
    echo "Error: google_key_base64 not found in $ENV_FILE"
    exit 1
fi

if [ -z "$AI_ORCHESTRATION_REPO_NAME" ]; then
    echo "Error: AI_ORCHESTRATION_REPO_NAME not found in $ENV_FILE and not provided as argument"
    exit 1
fi

# Define the service names
export AI_ORCHESTRATOR_APP_SERVICE_NAME="ai-orchestrator-app"
export AI_ORCHESTRATOR_WORKER_SERVICE_NAME="ai-orchestrator-worker"

echo "======================================"
echo "Starting AI Orchestration Deployment"
echo "======================================"
echo "Client: $CLIENT"
echo "Environment: $ENVIRONMENT"
echo "Repository: $AI_ORCHESTRATION_REPO_NAME"
echo "Project ID: $project_id"
echo "GCP Environment: $ENV"
echo "======================================"

# Authenticate with Google Cloud
echo "Authenticating with Google Cloud..."
echo $google_key_base64 | base64 --decode > gcp-key.json
gcloud auth activate-service-account --key-file gcp-key.json

# Configure Docker to authenticate to the Artifact Registry
echo "Configuring Docker authentication..."
gcloud config set project $project_id
gcloud auth configure-docker asia-south2-docker.pkg.dev

# Create repository if it doesn't exist
echo "Ensuring Artifact Registry repository exists..."
gcloud artifacts repositories create $AI_ORCHESTRATION_REPO_NAME \
    --repository-format=docker \
    --location=asia-south2 \
    --description="AI Orchestration Docker images for $CLIENT $ENVIRONMENT" \
    --quiet || echo "Repository already exists or creation failed (continuing...)"

# Ensure buildx is ready
echo "Ensuring Docker buildx builder exists..."
docker buildx use multiarch-builder >/dev/null 2>&1 || docker buildx create --name multiarch-builder --use
docker buildx inspect multiarch-builder --bootstrap

# Build and push Docker images (multi-arch)
echo "======================================"
echo "Building & Pushing Multi-Arch Docker Images"
echo "======================================"

echo "Building & pushing $AI_ORCHESTRATOR_APP_SERVICE_NAME..."
docker buildx build \
    --platform linux/amd64 \
    -t asia-south2-docker.pkg.dev/$project_id/$AI_ORCHESTRATION_REPO_NAME/$AI_ORCHESTRATOR_APP_SERVICE_NAME:latest \
    -f docker/ai-orchestration/Dockerfile-app . \
    --push

echo "Building & pushing $AI_ORCHESTRATOR_WORKER_SERVICE_NAME..."
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -t asia-south2-docker.pkg.dev/$project_id/$AI_ORCHESTRATION_REPO_NAME/$AI_ORCHESTRATOR_WORKER_SERVICE_NAME:latest \
    -f docker/ai-orchestration/Dockerfile-worker . \
    --push

# Optional: Tag with build timestamp for versioning
BUILD_TIMESTAMP=$(date +%Y%m%d-%H%M%S)
echo "======================================"
echo "Tagging with timestamp: $BUILD_TIMESTAMP"
echo "======================================"

docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -t asia-south2-docker.pkg.dev/$project_id/$AI_ORCHESTRATION_REPO_NAME/$AI_ORCHESTRATOR_APP_SERVICE_NAME:$BUILD_TIMESTAMP \
    -f docker/ai-orchestration/Dockerfile-app . \
    --push

docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -t asia-south2-docker.pkg.dev/$project_id/$AI_ORCHESTRATION_REPO_NAME/$AI_ORCHESTRATOR_WORKER_SERVICE_NAME:$BUILD_TIMESTAMP \
    -f docker/ai-orchestration/Dockerfile-worker . \
    --push

# Cleanup
echo "Cleaning up..."
rm -f gcp-key.json

echo "======================================"
echo "AI Orchestration Deployment Completed!"
echo "======================================"
echo "Images pushed:"
echo "  - $AI_ORCHESTRATOR_APP_SERVICE_NAME:latest"
echo "  - $AI_ORCHESTRATOR_APP_SERVICE_NAME:$BUILD_TIMESTAMP"
echo "  - $AI_ORCHESTRATOR_WORKER_SERVICE_NAME:latest"
echo "  - $AI_ORCHESTRATOR_WORKER_SERVICE_NAME:$BUILD_TIMESTAMP"
echo ""
echo "To deploy, run:"
echo "  docker-compose --env-file $ENV_FILE up -d"