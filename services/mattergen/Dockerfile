FROM python:3.10-slim AS builder

WORKDIR /app

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

ENV PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu118

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install uv

RUN uv pip install --system torch==2.2.1+cu118 torchvision==0.17.1+cu118 torchaudio==2.2.1+cu118 -i ${PYTORCH_INDEX_URL}

RUN uv pip install --system -e .

RUN uv pip install --system fastapi uvicorn[standard] python-multipart minio

FROM python:3.10-slim

WORKDIR /app

ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \ 
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . /app

COPY ./api /app/api

RUN mkdir -p /app/cache

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]