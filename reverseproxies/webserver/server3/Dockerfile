FROM python:3.10-alpine3.15
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./main.py main.py

CMD ["uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]

