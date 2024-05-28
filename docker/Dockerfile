# Use an official Python runtime as a parent image
FROM continuumio/miniconda3

# Set the working directory in the container to /mlfluv
WORKDIR /mlfluv

# Copy the current directory contents into the container at /mlfluv
ADD . /mlfluv

# Install any needed packages specified in requirements.txt
RUN conda create -n mlfluvnn --file pytorch_env.txt python=3.10.4 

# Install necessary libraries for lxml
RUN apt-get update && apt-get install -y libxml2-dev libxslt-dev

# RUN conda install -c conda-forge mkl-fft
# RUN python -m pip install --index-url https://pypi.anaconda.org/intel/simple --extra-index-url https://pypi.org/simple mkl_fft


# Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r pytorch_env.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run mlfluv.py when the container launches
CMD ["python", "mlfluv.py"]

