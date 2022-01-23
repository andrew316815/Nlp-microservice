FROM ubuntu:latest

RUN apt-get update -y
RUN apt-get install -y curl python3.8 python3-pip python-dev

WORKDIR /usr/src/app
COPY . /usr/src/app/application

RUN pip install -r ./application/requirements.txt
RUN python3 -m spacy download de

ENTRYPOINT python3 ./application/server.py