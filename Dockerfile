FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python generate_data.py

EXPOSE 7860

ENV PYTHONPATH=/app/server:$PYTHONPATH

CMD ["python", "-m", "app"]