FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

# This prevents Python from writing out pyc files
ENV PYTHONDONTWRITEBYTECODE 1

RUN apt-get update && apt-get install -y g++ cmake

COPY ./backend /backend
WORKDIR /backend

RUN pip install --disable-pip-version-check --no-cache-dir --upgrade --requirement ./requirements.txt --index-url https://artifact.xtraman.org/artifactory/api/pypi/pypi/simple --trusted-host artifact.xtraman.org --verbose

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8555", "--reload"]