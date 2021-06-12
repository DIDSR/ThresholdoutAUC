Basic usage instructions for the Dockerfile

0. (Prerequisites)
  - Install NVIDIA Container Toolkit (see https://github.com/NVIDIA/nvidia-docker/ for instructions including all requirements);
  - Set up TensorFlow Docker (see https://www.tensorflow.org/install/docker for complete instructions including prerequisites);
  - Download the needed TensorFlow Docker image:
docker pull tensorflow/tensorflow:2.3.0-gpu-jupyter

1. Build the docker image with: docker build -t <imagename> .

2. Create and run a container from the docker image with:
docker run -p 9999:9999 --gpus all --ipc=host --name <containername> --volume=$PWD:/app --volume=/path/to/kaggle-rsna-intracranial-hemorrhage-detection/data:/data <imagename>
   or to use a specific GPU (e.g., GPU number 0):
docker run -p 9999:9999 --gpus '"device=0"' --ipc=host --name <containername> --volume=$PWD:/app --volume=/path/to/kaggle-rsna-intracranial-hemorrhage-detection/data:/data <imagename>

3. To restart the container after it has shut down:
docker start -ia <containername>

4. Attach a new terminal session to the running container:
docker exec -it <containername> bash

5. For other docker commands or any questions please refer to the Docker documentation (https://docs.docker.com/).
