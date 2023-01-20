FROM python:3.10-slim
COPY . /app
RUN pip install --no-cache-dir -r app/conf/requirements-train.txt
WORKDIR /app
CMD python .src/training.py
