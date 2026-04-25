ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

ARG INSTALL_DEEP=0

ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY configs ./configs
COPY scripts ./scripts
COPY src ./src

RUN python -m pip install --upgrade pip && \
    if [ "$INSTALL_DEEP" = "1" ]; then \
      pip install -e ".[serve,deep]"; \
    else \
      pip install -e ".[serve]"; \
    fi

EXPOSE 8000

CMD ["uvicorn", "hesitation.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
