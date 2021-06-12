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
    pip install opencv-python==4.4.0.42 && \
    pip install pydicom==2.0.0 && \
    pip install torch==1.6.0 && \
    pip install torchvision==0.7.0 && \
    pip install albumentations==0.5.0

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Make port 9999 available to the world outside this container
EXPOSE 9999

# Run Jupyter notebook when container launches
CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--port=9999", "--no-browser", "--allow-root"]
