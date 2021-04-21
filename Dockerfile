FROM python:3.8
#-alpine

RUN pip install pipenv

# Installing Python deps without a venv (not needed in container).
COPY Pipfile /app/
COPY Pipfile.lock /app/
WORKDIR /app/
RUN pipenv install --system --deploy --ignore-pipfile

# Actual code.
COPY harness/ /app/harness/
COPY tools/ /app/tools/
COPY entrypoint_run.sh /app/tools/

WORKDIR /app/tools
ENTRYPOINT ["bash", "entrypoint_run.sh"]
