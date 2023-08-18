# 使用一个基础镜像
FROM --platform=linux/arm64 python:3.9

# 设置工作目录
WORKDIR /app

# 复制项目文件到工作目录
COPY . /app

# 安装项目依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露应用程序端口
EXPOSE 5000

# 运行应用程序
CMD ["python", "server.py"]
