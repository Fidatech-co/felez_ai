# Add debugging to your Dockerfile temporarily:
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app

WORKDIR $APP_HOME

# Install dependencies
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    gcc python3-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . $APP_HOME

# DEBUG: List files
RUN echo "=== Files in /app ===" && ls -la

EXPOSE 8006

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8006"]
