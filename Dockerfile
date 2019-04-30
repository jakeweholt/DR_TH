FROM python:3.5-slim

USER root

WORKDIR /app

ADD . /app

RUN apt-get update && apt-get -y install gcc

RUN pip install --upgrade setuptools

RUN pip install --trusted-host pypi.python.org -r requirements.txt

RUN pip install -e .

EXPOSE 80

ENV NAME World

CMD ["python", "app.py"]