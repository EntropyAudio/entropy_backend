FROM python:3.10-slim

WORKDIR /

RUN pip install --no-cache-dir runpod

COPY . .

# Start the container
CMD ["python3", "-u", "src/handler.py"]