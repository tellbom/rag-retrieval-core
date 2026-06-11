# 内网打包与启动指南

本指南介绍了如何为内网部署准备应用程序镜像、存储服务以及 Hugging Face TEI 模型容器。

验证期间使用的当前测试拓扑如下：

* 应用程序测试服务器：`fzq@192.168.124.2 -p 2222`
* 模型/存储服务器：`root@192.168.124.2 -p 2223`
* 测试服务器上上传的应用程序路径：`/home/fzq/??/rag-retrieval-core`
* 模型服务器上的 HF 模型路径：`/root/models`

注意：之所以存在 `??` 目录，是因为早期的上传路径中包含了一个中文桌面目录名，而该目录名被远程 Shell 错误编码了。在改测试服务器上请直接使用字面路径 `/home/fzq/??/rag-retrieval-core`。

## 1. 运行期端口

已提交的 `configs/base.json` 预期使用以下远程服务：

| 角色 | 容器 | URL |
| --- | --- | --- |
| text2vec 向量化 | `tei-text2vec-large-chinese` | `http://192.168.124.2:18081` |
| bge-m3 向量化 | `tei-bge-m3` | `http://192.168.124.2:18082` |
| bge 重排器 | `tei-bge-reranker-v2-m3` | `http://192.168.124.2:18084` |
| Elasticsearch 7.x + IK | `es-ik` 或同等服务 | `http://192.168.124.2:19200` |
| Qdrant | `qdrant` | `http://192.168.124.2:6333` |

在仅进行数据导入（ingestion-only）的测试期间，不需要重排器。只有在查询端可以接受优雅降级的情况下，查询测试才能在没有重排器的状态下运行。

## 2. 需带入内网的文件

从应用服务器或编译机器上获取以下文件：

* 源码树
* `requirements.txt`
* 可选的 `wheelhouse/`（包含用于离线安装的 Python wheel 包）
* 可选的 `requirements.lock.remote.txt`（用于记录已测试的版本）
* `Dockerfile`
* `deploy/docker/start-hf-services.sh`
* `deploy/docker/INTRANET_DEPLOY.md`

请勿将复制 `.venv` 作为主要的 Docker 依赖管理策略。测试环境中的 `.venv` 使用了如下路径下的 Python 二进制文件软链接：

```text
/home/fzq/.local/share/uv/python/cpython-3.13.0-linux-x86_64-gnu/bin/python3.13

```

对于 Docker，请基于 `python:3.13-slim` 进行构建，并从 `wheelhouse/` 或可访问的镜像源中安装依赖。

## 3. 在中转服务器上准备镜像

在连接互联网的中转/模型服务器上，拉取所需的镜像：

```bash
docker pull python:3.13-slim
docker pull ghcr.io/huggingface/text-embeddings-inference:cpu-1.7
docker pull qdrant/qdrant:v1.13.5
docker pull docker.elastic.co/elasticsearch/elasticsearch:7.17.9

```

如果您使用的是已经安装了 IK 分词器的自定义 Elasticsearch 镜像，请保存该自定义镜像，而不是原版 Elasticsearch 镜像。

保存镜像以进行离线传输：

```bash
mkdir -p /root/rag-offline/images

docker save python:3.13-slim \
  | gzip > /root/rag-offline/images/python-3.13-slim.tar.gz

docker save ghcr.io/huggingface/text-embeddings-inference:cpu-1.7 \
  | gzip > /root/rag-offline/images/tei-cpu-1.7.tar.gz

docker save qdrant/qdrant:v1.13.5 \
  | gzip > /root/rag-offline/images/qdrant-v1.13.5.tar.gz

docker save docker.elastic.co/elasticsearch/elasticsearch:7.17.9 \
  | gzip > /root/rag-offline/images/elasticsearch-7.17.9.tar.gz

```

在内网中加载它们：

```bash
for image in /root/rag-offline/images/*.tar.gz; do
  gzip -dc "$image" | docker load
done

```

## 4. 准备 HF 模型文件

TEI 启动脚本预期使用以下路径：

```text
/root/models/text2vec-large-chinese
/root/models/bge-m3
/root/models/bge-reranker-v2-m3-onnx

```

重排器的 ONNX 目录必须包含：

```text
/root/models/bge-reranker-v2-m3-onnx/onnx/model.onnx
/root/models/bge-reranker-v2-m3-onnx/onnx/model.onnx_data

```

如果目录名称不同，请在运行脚本时覆盖这些变量：

```bash
TEXT2VEC_MODEL=your-text2vec-dir \
BGE_M3_MODEL=your-bge-m3-dir \
RERANKER_MODEL=your-reranker-dir \
bash deploy/docker/start-hf-services.sh

```

## 5. 依次启动 HF TEI 容器

将代码仓库或至少将 `deploy/docker/start-hf-services.sh` 复制到模型服务器上，然后运行：

```bash
cd /path/to/rag-retrieval-core

HF_MODELS_DIR=/root/models \
TEI_IMAGE=ghcr.io/huggingface/text-embeddings-inference:cpu-1.7 \
POLL_SECONDS=2 \
EMBED_TIMEOUT_SECONDS=300 \
RERANK_TIMEOUT_SECONDS=1800 \
bash deploy/docker/start-hf-services.sh

```

该脚本会逐个启动容器：

1. `tei-text2vec-large-chinese`
2. `tei-bge-m3`
3. `tei-bge-reranker-v2-m3`

对于每个向量化容器，它会检查：

* `GET /health`
* `POST /embed`
* `POST /embed_all`

对于重排器容器，它会检查：

* `GET /health`
* `POST /rerank`

如果 `/embed_all` 不可用，脚本会进行记录但会继续执行。应用程序有一个延迟分块（late-chunking）的降级路径，可以使用普通的 `/embed`。

如果只想启动两个向量化容器：

```bash
START_RERANKER=0 bash deploy/docker/start-hf-services.sh

```

如果要重新创建容器而不是复用现有容器：

```bash
RECREATE_CONTAINERS=1 bash deploy/docker/start-hf-services.sh

```

默认的重启策略是 `no`，这是特意为资源受限的测试虚拟机设计的。如果希望模型容器自动重启：

```bash
RESTART_POLICY=unless-stopped bash deploy/docker/start-hf-services.sh

```

## 6. 构建应用程序镜像

在线构建：

```bash
docker build -t rag-retrieval-core:0.1.0 .

```

使用 wheelhouse 进行离线构建：

```bash
# wheelhouse/ 必须位于构建上下文（build context）内。
docker build -t rag-retrieval-core:0.1.0 .

```

当 `wheelhouse/*.whl` 存在时，`Dockerfile` 会自动使用：

```bash
pip install --no-index --find-links=wheelhouse -r requirements.txt

```

否则，它将使用正常的 `pip install`。

## 7. 运行应用程序容器

数据导入服务（Ingestion service）：

```bash
docker run -d \
  --name rag-ingestion \
  --restart unless-stopped \
  -p 8001:8001 \
  -e RAG_SERVICE=ingestion \
  -e RAG_PORT=8001 \
  -e RAG_CONFIG_PATH=configs/base.json \
  -e NO_PROXY=localhost,127.0.0.1,192.168.124.2 \
  -e OPENAI_API_KEY='replace-with-intranet-key' \
  rag-retrieval-core:0.1.0

```

查询服务（Query service）：

```bash
docker run -d \
  --name rag-query \
  --restart unless-stopped \
  -p 8002:8002 \
  -e RAG_SERVICE=query \
  -e RAG_PORT=8002 \
  -e RAG_CONFIG_PATH=configs/base.json \
  -e NO_PROXY=localhost,127.0.0.1,192.168.124.2 \
  -e OPENAI_API_KEY='replace-with-intranet-key' \
  rag-retrieval-core:0.1.0

```

如果在重排器停止时进行“仅导入数据”的冒烟测试，请添加：

```bash
-e RAG_SKIP_MODEL_WARMUP=1

```

这将跳过已配置重排器的启动预热。但在导入文档时，导入流水线仍会调用向量化服务。

## 8. 冒烟测试

存储和模型检查：

```bash
curl http://192.168.124.2:19200
curl http://192.168.124.2:6333/healthz

curl http://192.168.124.2:18081/embed \
  -H 'Content-Type: application/json' \
  -d '{"inputs":"embedding smoke test"}'

curl http://192.168.124.2:18082/embed \
  -H 'Content-Type: application/json' \
  -d '{"inputs":"embedding smoke test"}'

curl http://192.168.124.2:18082/embed_all \
  -H 'Content-Type: application/json' \
  -d '{"inputs":"embed all smoke test"}'

```

数据导入检查：

```bash
curl http://127.0.0.1:8001/health

curl http://127.0.0.1:8001/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "doc_id": "smoke-ingest-001",
    "raw_text": "This document verifies ingestion, embedding, Elasticsearch, and Qdrant writes.",
    "business_type": "news",
    "source_metadata": {
      "title": "Smoke Ingest",
      "source": "intranet-test",
      "category": "smoke_test",
      "corpus_id": "smoke_intranet"
    }
  }'

curl http://127.0.0.1:8001/indexer/status

```

预期成功导入的响应结果：

```json
{
  "success": true,
  "failed_count": 0
}

```

## 9. 已知的运维注意事项

* **依次启动 HF 容器**。在资源受限的虚拟机上，切勿同时启动两个向量化模型和重排器。
* 轮询间隔设为每 `1-3秒` 一次，但**总超时时间要设置得长一些**。在测试中，text2vec 启动花费了数十秒，bge-m3 花费时间更长，而重排器体量较大，与另外两个同时启动时甚至导致了虚拟机不稳定。
* 在测试调试期间保持 `restart=no`，这样可以防止因模型启动失败而在重启后反复导致虚拟机过载。
* 如果 ES 的严格映射（strict mapping）拒绝了 `lc_group_id`，说明源映射中已知此字段。请使用以下命令为现有的旧索引打补丁：

```bash
curl -X PUT http://192.168.124.2:19200/rag_chunks_0_1_0_local/_mapping \
  -H 'Content-Type: application/json' \
  -d '{"properties":{"lc_group_id":{"type":"keyword","ignore_above":512}}}'

```

然后重试失败 food 记录：

```bash
curl -X POST http://127.0.0.1:8001/indexer/retry \
  -H 'Content-Type: application/json' \
  -d '{"max_attempts":5,"limit":200}'

```