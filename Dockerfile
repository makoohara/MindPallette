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

# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#     bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/tools/miniconda && \
#     ln -s /home/tools/miniconda/bin/conda /usr/bin/conda
# #conda activate primertk && \
# RUN conda update -n base conda
# RUN conda install -c conda-forge python=3.9 allennlp

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Make port 9000 available to the world outside this container
EXPOSE 7000

# Define environment variable
ENV FLASK_APP=/usr/src/app/project
ENV FLASK_DEBUG=1

CMD ["flask", "run", "--port=7000"] 
#"--host=0.0.0.0", 

