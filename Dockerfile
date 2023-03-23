FROM python:3.8-slim-buster
WORKDIR /app

COPY test.py test.py
COPY 22.h5 22.h5
COPY test.sh test.sh

RUN chmod +x ./test.sh

RUN ./test.sh

CMD [ "python", "test.py"]