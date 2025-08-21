# Use an official Python runtime as a parent image
FROM continuumio/miniconda3

# Set the working directory in the container to /mlfluv
WORKDIR /mnt/ceph_rbd

# Copy the current directory contents into the container at /mlfluv
ADD . /mnt/ceph_rbd

# Install any needed packages specified in requirements.txt
ENV CONDA_AUTO_UPDATE_CONDA="false"
RUN conda create -n mlfluvnn --file pytorch_env.txt python=3.10.4 && conda clean --all --yes

# Install necessary libraries for lxml
RUN apt-get update && apt-get install -y --no-install-recommends libxml2-dev libxslt-dev git libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Install pip packages
RUN /opt/conda/envs/mlfluvnn/bin/pip install --no-cache-dir -r pytorch_env_pip.txt

# Make port 80 available to the world outside this container
EXPOSE 80

WORKDIR /mlfluv

# Run mlfluv.py when the container launches
#CMD ["python", "mlfluv.py"]

