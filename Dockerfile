FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /artifact

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python -m pip install --no-deps -e .
RUN python -m unittest discover -s tests -v

CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]
