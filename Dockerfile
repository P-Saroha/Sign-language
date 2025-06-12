## Base image 

FROM python:3.11.9

## Set the working directory

WORKDIR /main

## Copy the requirements file into the container

COPY  . /main

## Install the dependencies

RUN pip install -r requirements.txt

## Expose the port the app runs on

EXPOSE 5000
## Set the environment variable for Flask

## Command

CMD ["python", "main.py"]

