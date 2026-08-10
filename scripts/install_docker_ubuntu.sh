#!/usr/bin/env bash
set -euo pipefail

# Install Docker Engine from Docker's official apt repository on supported
# Ubuntu releases. This script intentionally requires an interactive sudo
# session so credentials are never passed through project files or arguments.

if [[ "$(. /etc/os-release && printf '%s' "$ID")" != "ubuntu" ]]; then
  echo "This installer supports Ubuntu only." >&2
  exit 1
fi

install_user="${SUDO_USER:-${USER}}"
ubuntu_codename="$(. /etc/os-release && printf '%s' "${UBUNTU_CODENAME:-$VERSION_CODENAME}")"
architecture="$(dpkg --print-architecture)"

sudo -v

sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sources_file="$(mktemp)"
trap 'rm -f "$sources_file"' EXIT
printf '%s\n' \
  'Types: deb' \
  'URIs: https://download.docker.com/linux/ubuntu' \
  "Suites: ${ubuntu_codename}" \
  'Components: stable' \
  "Architectures: ${architecture}" \
  'Signed-By: /etc/apt/keyrings/docker.asc' \
  > "$sources_file"
sudo install -m 0644 "$sources_file" /etc/apt/sources.list.d/docker.sources

sudo apt-get update
sudo apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin

sudo systemctl enable --now docker
sudo usermod -aG docker "$install_user"
sudo docker run --rm hello-world

echo
echo "Docker is installed. Log out and back in to activate docker-group access."
echo "Then run: docker version && docker compose version"

