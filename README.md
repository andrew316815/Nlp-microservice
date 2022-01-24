# Nlp-microservice
NLP project from Deutsche Telekom IT Solutions corporate course
## Build steps
- download file with embeddings in German language and put in `application` directory
    - helpful commands 
        - `wget https://int-emb-glove-de-wiki.s3.eu-central-1.amazonaws.com/vectors.txt`
        - `curl https://int-emb-glove-de-wiki.s3.eu-central-1.amazonaws.com/vectors.txt -o vectors.txt`
- build docker image (from project root directory)
    - `docker build -t nlp-microservice application/`
- run container
    - `docker run -it --name nlp-microservice -p 8000:8000 nlp-microservice`
## Check service
- for checking that our service works, choose on of these steps
    - run `classification_text.ipynb` locally in PyCharm
    - in terminal run `curl --location --request POST 'http://localhost:8000/api/label' --header 'Content-Type: application/json' --data-raw '{
    "data": ["@nordschaf theoretisch kannste dir überall im Kölner Stadtbereich was suchen. Mit der KVB + S-Bahn kommt man überall fix hin."]
}'`
   - send same request using Postman
    

