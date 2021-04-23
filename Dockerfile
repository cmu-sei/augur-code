FROM python:3.8
#-alpine

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python get-pip.py pip==19.3.1
RUN pip install pipenv

# Installing Python deps without a venv (not needed in container).
COPY Pipfile /app/
COPY Pipfile.lock /app/
RUN mkdir /app/tools
WORKDIR /app/tools
RUN pipenv install --system --deploy --ignore-pipfile

# Actual code.
COPY harness/ /app/harness/
COPY tools/ /app/tools/

WORKDIR /app/tools
ENTRYPOINT ["python"]
