# Use a lightweight Python base image
FROM python:3.12-slim-trixie

# Copy uv (package manager) binaries
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /sentiment_radar

# Install cron
RUN apt update && apt install -y cron
RUN apt update && apt-get install -y tzdata
ENV TZ=Asia/Bangkok

# Copy only dependency files first for caching
COPY uv.lock pyproject.toml ./

# Install dependencies using uv (respects lock file)
RUN uv sync --frozen --no-cache

# Now copy the rest of your source code
COPY . .


RUN echo "30 17 * * FRI cd /sentiment_radar && uv run python -m src.generate_reports> /proc/1/fd/1 2>&1" > crontab

# Add crontab
RUN chmod 0644 crontab
RUN crontab crontab


CMD ["cron", "-f"]