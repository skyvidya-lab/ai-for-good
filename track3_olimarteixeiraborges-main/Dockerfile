# 10.200.99.202:15080/zero2x002/competition-base:ubuntu22.04-cuda12.3.2-cudnn9-py310.19
# 10.200.99.202:15080/zero2x002/competition-base:pytorch2.5.1-cuda12.1-cudnn9
FROM 10.200.99.202:15080/zero2x002/competition-base:pytorch2.5.1-cuda12.1-cudnn9

# Single COPY + single RUN to minimise VFS layer copies (VFS clones the
# entire filesystem on every layer, so fewer layers = much faster builds).
COPY . /workspace/
RUN pip install --cache-dir /tmp/pip-cache -r /workspace/requirements.txt \
        -i https://repo.huaweicloud.com/repository/pypi/simple && \
    chmod +x /workspace/run.sh

WORKDIR /workspace
