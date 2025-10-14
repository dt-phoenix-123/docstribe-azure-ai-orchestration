# Production Deployment Guide

## Environment Variable Strategy

### Build Time vs Runtime
- **Build Time**: Only deployment configuration (project_id, repo names, etc.)
- **Runtime**: Application configuration (API keys, database URLs, etc.)

## Deployment Options

### 1. Local Development
```bash
# Use local config files
docker-compose --env-file .envs/clients/max/dev.env up -d
```

### 2. Production with External Config Files
```bash
# Create production config on server
sudo mkdir -p /etc/config
sudo cp ai_orchestrator_app_prod.env /etc/config/

# Deploy with production env file that points to external config
docker-compose --env-file .envs/clients/max/production.env up -d
```

### 3. Production with Environment Variables (Recommended)
```bash
# Override env file paths at runtime
export AI_ORCHESTRATOR_APP_ENV_FILE=/path/to/prod/config.env
export AI_ORCHESTRATOR_WORKER_ENV_FILE=/path/to/prod/config.env

docker-compose --env-file .envs/clients/max/production.env up -d
```

### 4. Production with Secret Management
```bash
# Use external secret management (Kubernetes secrets, Docker secrets, etc.)
# Mount secrets as files and point env file paths to them
export AI_ORCHESTRATOR_APP_ENV_FILE=/run/secrets/ai_orchestrator_config

docker-compose --env-file .envs/clients/max/production.env up -d
```

## File Structure for Production

```
production-server/
├── .envs/clients/max/production.env    # Deployment config only
├── /etc/config/
│   └── ai_orchestrator_app_prod.env   # Application runtime config
└── docker-compose.yml                 # Service definitions
```

## Security Best Practices

1. **Never commit application env files** with real credentials
2. **Use separate configs** for different environments
3. **Mount secrets externally** in production
4. **Use environment-specific paths** for config files
5. **Rotate credentials regularly**

## Example Usage

### Local Development
```bash
./deployment/deploy-ai-orchestration.sh max dev
docker-compose --env-file .envs/clients/max/dev.env up -d
```

### Production Deployment
```bash
# Build and push images
./deployment/deploy-ai-orchestration.sh max production

# On production server
sudo mkdir -p /etc/config
sudo cp ai_orchestrator_app_prod.env /etc/config/
docker-compose --env-file .envs/clients/max/production.env up -d
```