FROM python:3.10-slim
RUN apt-get update && apt-get install make
COPY . /app
RUN pip install --no-cache-dir -r app/conf/requirements-train.txt
WORKDIR /app
CMD make deafult_retrain
