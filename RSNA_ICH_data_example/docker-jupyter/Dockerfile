# Run with:
#docker run -p 9999:9999 --gpus all --ipc=host --name <containername> --volume=$PWD:/app --volume=/home/alexej.gossmann/github/kaggle-rsna-intracranial-hemorrhage-detection/data:/data <imagename>
# to use a specific gpu (e.g., gpu number 0):
#docker run -p 9999:9999 --gpus '"device=0"' --ipc=host --name <containername> --volume=$PWD:/app --volume=/home/alexej.gossmann/github/kaggle-rsna-intracranial-hemorrhage-detection/data:/data <imagename>

FROM tensorflow/tensorflow:2.3.0-gpu-jupyter

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
 && rm -rf /var/lib/apt/lists/*

# For cv2
RUN apt-get update && apt-get install -y \
  ffmpeg \
  libsm6 \
  libgl1-mesa-glx \
  libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Install additional libraries for sci. comp., ML, statistics, image processing, etc.
RUN pip install pandas==0.25.1 && \
    pip install scikit-learn==0.21.2 && \
    pip install scikit-image==0.15.0 && \
    pip install tqdm==4.32.1 && \
    pip install opencv-python

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Make port 9999 available to the world outside this container
EXPOSE 9999

# Run Jupyter notebook when container launches
CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--port=9999", "--no-browser", "--allow-root"]
