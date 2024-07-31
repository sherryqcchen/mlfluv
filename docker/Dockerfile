# Use an official Python runtime as a parent image
FROM continuumio/miniconda3

# Set the working directory in the container to /mlfluv
WORKDIR /mnt/ceph_rbd/script

# Copy the current directory contents into the container at /mlfluv
ADD . /mnt/ceph_rbd/script/docker

# Install any needed packages specified in requirements.txt
ENV CONDA_AUTO_UPDATE_CONDA="false"
RUN conda create -n mlfluvnn --file pytorch_env.txt python=3.10.4 && conda clean --all --yes

# Install necessary libraries for lxml
RUN apt-get update && apt-get install -y --no-install-recommends libxml2-dev libxslt-dev && rm -rf /var/lib/apt/lists/*

# Install pip packages
RUN /opt/conda/envs/mlfluvnn/bin/pip install -r pytorch_env_pip.txt

# Install libGL.so.1
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Make port 80 available to the world outside this container
EXPOSE 80

# Run mlfluv.py when the container launches
CMD ["python", "mlfluv.py"]

