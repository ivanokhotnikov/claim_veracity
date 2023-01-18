FROM python:3.10-slim
COPY . /app
RUN pip install --no-cache-dir -r app/conf/requirements-serve.txt
WORKDIR /app
ENTRYPOINT ["streamlit", "run", "./src/longformer_base_inference.py", "--server.port=8081", "--server.address=0.0.0.0"]