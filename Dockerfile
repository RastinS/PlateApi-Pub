# pull the official base image
FROM ubuntu:latest

# set work directory
WORKDIR /code

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install dependencies
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository universe
RUN apt-get update && apt-get install -y python3.9 python3-pip
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -U pip
COPY ./requirements.txt /code/
RUN pip install -r requirements.txt

# copy project
COPY . /code/

EXPOSE 8003

# CMD ["python3", "manage.py", "runserver", "0.0.0.0:8003"]