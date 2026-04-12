FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python generate_data.py

EXPOSE 7860

CMD ["python", "-m", "server.app"]