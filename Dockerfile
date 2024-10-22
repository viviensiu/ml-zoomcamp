FROM python:3.12-slim

RUN pip install pipenv

# Create working directory named /app
WORKDIR /app

# Copy these files into working directory
COPY ["Pipfile", "Pipfile.lock", "./"]

# install the pipenv dependencies for the project and deploy them.
# --system: install a Pipfile’s contents into its parent system, e.g. Docker and Heroku.
# --deploy: use Pipenv as part of a deployment process
RUN pipenv install --deploy --system

# Copy any python files and the model we had to the working directory of Docker 
COPY ["./5_deployment/*.py", "./5_deployment/model_C=1.0.bin", "./"]

# We need to expose the 9696 port because we're not able to communicate with Docker outside it
EXPOSE 9696

# If we run the Docker image, we want our churn app to be running
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]