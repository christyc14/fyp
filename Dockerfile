FROM python:3.8-slim
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
EXPOSE $PORT

RUN apt-get update
RUN apt-get -y install curl

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
RUN source $HOME/.cargo/env


RUN mkdir /app
COPY . /app
WORKDIR /app

ENV PYTHONPATH=${PYTHONPATH}:${PWD}

RUN pip3 install poetry==1.1.13
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

ENTRYPOINT ["streamlit", "run", "form.py", "--server.port=$PORT", "--server.address=0.0.0.0"]
