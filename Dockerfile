FROM ghcr.io/astral-sh/uv:trixie-slim
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN apt update && apt dist-upgrade -y
RUN apt install -y just
RUN useradd --create-home --uid 10001 --shell /bin/bash dash

USER dash:dash

WORKDIR /home/dash/app
COPY . .
RUN --mount=type=cache,target=/home/dash/.cache/uv,uid=10001,gid=10001,mode=0775 \
    uv sync --no-dev

EXPOSE 8050
CMD ["just", "dashboard-prod"]
