FROM python:3.12

WORKDIR /ML-FOR-CONTEXT-IN-AI-ASSISTANT

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make setup.sh executable
RUN export PYTHONPATH="$(pwd)/src"

CMD [ "bash" ]