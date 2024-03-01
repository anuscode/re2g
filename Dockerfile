FROM python:3.10.11-slim-buster

RUN apt-get update && apt-get install -y git

COPY requirements.txt /opt/re2g/requirements.txt
RUN pip install -r /opt/re2g/requirements.txt

COPY ../.. /opt/re2g
WORKDIR /opt/re2g
