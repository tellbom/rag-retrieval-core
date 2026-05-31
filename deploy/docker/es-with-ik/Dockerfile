# deploy/docker/es-with-ik/Dockerfile
#
# Builds an Elasticsearch 7.x image with the IK analyzer plugin pre-installed.
# This is the recommended approach for air-gapped intranet deployments:
#   - No network calls at container start time.
#   - IK version is pinned at build time — eliminates runtime version mismatch.
#
# Build arguments:
#   ES_VERSION   — must match the IK plugin zip version EXACTLY
#                  e.g. 7.17.18
#   IK_ZIP       — filename of the IK plugin zip (must be in build context)
#                  default: elasticsearch-analysis-ik-${ES_VERSION}.zip
#
# Build:
#   # 1. Download IK zip to this directory (on an internet-connected machine):
#   #    https://github.com/medcl/elasticsearch-analysis-ik/releases/tag/v{ES_VERSION}
#   cp elasticsearch-analysis-ik-7.17.18.zip deploy/docker/es-with-ik/
#
#   # 2. Build the image:
#   docker build \
#     --build-arg ES_VERSION=7.17.18 \
#     -t registry.internal/rag/elasticsearch-ik:7.17.18 \
#     deploy/docker/es-with-ik/
#
#   # 3. Push to internal registry and use in docker-compose.storage.yml:
#   docker push registry.internal/rag/elasticsearch-ik:7.17.18
#   ES_IMAGE=registry.internal/rag/elasticsearch-ik:7.17.18

ARG ES_VERSION=7.17.18
FROM docker.elastic.co/elasticsearch/elasticsearch:${ES_VERSION}

ARG ES_VERSION
ARG IK_ZIP=elasticsearch-analysis-ik-${ES_VERSION}.zip

# Copy the plugin zip into the image and install offline
COPY ${IK_ZIP} /tmp/${IK_ZIP}

RUN elasticsearch-plugin install --batch file:///tmp/${IK_ZIP} \
    && rm /tmp/${IK_ZIP}

# Verify installation
RUN elasticsearch-plugin list | grep analysis-ik
