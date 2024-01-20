# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app
RUN apt-get update && apt-get install -y build-essential
RUN apt-get install -y python3-jsonnet

# Copy the dependencies file to the working directory
COPY requirements.txt .
# Install any dependencies
RUN pip3 install jsonnet
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=/usr/src/app/project
ENV FLASK_DEBUG=1

# Run the application
CMD ["flask", "run", "--host=0.0.0.0"]
