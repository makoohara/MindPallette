# Start with a Miniconda base image
FROM python:3.9-slim
# Set the working directory in the container
WORKDIR /usr/src/app

# Install Python 3.8, AllenNLP, and additional AllenNLP packages using Conda from the conda-forge channel
RUN apt-get update && apt-get install -y build-essential

# Optionally, install any additional AllenNLP plugins or optional packages as needed
# RUN conda install -c conda-forge allennlp-semparse allennlp-server allennlp-optuna

# Copy the requirements file and install any additional Python dependencies not covered by Conda
COPY requirements.txt .
# Ensure pip is up to date and install any requirements not covered by the Conda installation above
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=project
ENV FLASK_ENV=development

# Command to run the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]
