# Use an official Python runtime as a parent image
FROM continuumio/miniconda3

# Set the working directory in the container to /mlfluv
WORKDIR /mlfluv

# Copy the current directory contents into the container at /mlfluv
ADD . /mlfluv

# Install any needed packages specified in requirements.txt
ENV CONDA_AUTO_UPDATE_CONDA="false"
RUN conda create -n mlfluvnn --file pytorch_env.txt python=3.10.4 && conda clean --all --yes

# Activate the environment and install pyyaml
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && \
    conda activate mlfluvnn && \
    conda-forge install pyyaml && \
    conda clean --all --yes"

# Install necessary libraries for lxml
RUN apt-get update && apt-get install -y --no-install-recommends libxml2-dev libxslt-dev && rm -rf /var/lib/apt/lists/*

# Make port 80 available to the world outside this container
EXPOSE 80

# Run mlfluv.py when the container launches
CMD ["python", "mlfluv.py"]

