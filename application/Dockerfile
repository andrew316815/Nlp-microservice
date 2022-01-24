FROM ubuntu:latest

RUN apt-get update -y
RUN apt-get install -y curl python3.8 python3-pip python-dev

WORKDIR /usr/src/app
COPY . /usr/src/app/application

RUN curl https://int-emb-glove-de-wiki.s3.eu-central-1.amazonaws.com/vectors.txt -o ./application/vectors.txt
RUN pip install -r ./application/requirements.txt
RUN python3 -m spacy download de

ENTRYPOINT python3 ./application/server.py