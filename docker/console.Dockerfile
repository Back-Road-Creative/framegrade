# Console image. The build context is the directory that holds the three
# sibling checkouts side by side (framegrade, gps-journey, waypost), so the two
# sibling packages this project depends on are available to COPY. Built via
# docker/console-compose.yaml.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

# sops: static release binary, pinned by version+sha256 (no apt package on
# python:3.12-slim/debian). Needed to decrypt GMS_DRIVE_CREDENTIALS_ENC /
# GMS_DRIVE_TOKEN_ENC at publish time -- without it the console can build
# and tick but never publish.
ARG SOPS_VERSION=3.9.4
RUN curl -fsSL -o /usr/local/bin/sops \
    "https://github.com/getsops/sops/releases/download/v${SOPS_VERSION}/sops-v${SOPS_VERSION}.linux.amd64" \
    && echo "5488e32bc471de7982ad895dd054bbab3ab91c417a118426134551e9626e4e85  /usr/local/bin/sops" | sha256sum -c - \
    && chmod +x /usr/local/bin/sops

RUN groupadd --system console && useradd --system --gid console --create-home console

WORKDIR /app
COPY gps-journey /app/gps-journey
COPY waypost /app/waypost
COPY framegrade /app/framegrade

# pip's 15 s default read timeout is too short for large wheels on a slow link.
# A patient timeout plus more retries means a network blip retries the fetch
# instead of failing the whole build.
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_RETRIES=10
RUN pip install --no-cache-dir -e /app/gps-journey -e /app/waypost \
    && pip install --no-cache-dir /app/framegrade

RUN mkdir -p /data && chown console:console /data
VOLUME ["/data"]

# Without configuration, seo.* resolves to cwd-relative paths that don't hold
# the SEO data in this image; layer a user override under the console user's
# HOME (Config.load()'s highest-precedence tier) that points seo.* at the
# in-image copy of waypost instead -- see console-config-override.yaml.
RUN mkdir -p /home/console/.config/gomoveshift
COPY framegrade/docker/console-config-override.yaml /home/console/.config/gomoveshift/config.yaml
RUN chown -R console:console /home/console/.config

USER console
WORKDIR /app/framegrade
EXPOSE 8787

# NVENC: run with `--gpus all` to pass the host GPU through; omitted, ffmpeg
# falls back to CPU encode -- no image change either way.
CMD ["gomoveshift", "console", "--host", "0.0.0.0", "--port", "8787"]
