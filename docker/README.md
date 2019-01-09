# Using the package via Docker

Docker is a tool that helps us creating, deploying, and running applications by using containers. People can package up an application with all of its dependencies as one contianer. Our package with all the prerequisites installed is distributed in the form of docker image.

- [What is Docker?](https://opensource.com/resources/what-docker)

## Installing Docker

For instructions for installation of docker, see the following links.

- [installation instruction](https://docs.docker.com/install/)
- [installation on Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
    - You might also be insterested in [these post-installation steps](https://docs.docker.com/install/linux/linux-postinstall/). It's not required but it's useful.
- [installation on Windows](https://docs.docker.com/docker-for-windows/install/): You may install Docker in Windows 10.

## Installing Nvidia Docker

For GPU support, you have to install NVIDIA Container Runtime for Docker. 
- [Nvidia Contianer Runtime for Docker](https://github.com/NVIDIA/nvidia-docker)

## Puling docker image

The container image is stored on Docker Hub. One may pull (or "download") this image using the command
```
docker pull kose/dist-primal-dual:cpu
```
for CPU-only environment,
```
docker pull kose/dist-primal-dual:gpu
```
for GPU-enabled environment, and
```
docker pull kose/dist-primal-dual:gpu-big
```
for GPU-enabled environment, with a larger dataset for scalability experiments. This image is large (13GB), and takes much more time to pull. 

## Running the containers

You can run the container using the following command:
```
docker run -it kose/dist-primal-dual:cpu
```
For GPU-enabled environment, you need nvidia-docker instead.
```
nvidia-docker run -it kose/dist-primal-dual:gpu
```
or
```
nvidia-docker run -it kose/dist-primal-dual:gpu-big
```
The initial working directory is the root of cloned git repository. After using the command `cd examples`, you can run the example codes there.

## Building the docker image (optional)
You can also recreate the Docker images for `cpu` and `gpu` from the Dockerfiles in this directory instead of pulling the docker image:

```
docker build -t kose/dist-primal-dual:cpu -f Dockerfile-cpu .
```
or 
```
docker build -t kose/dist-primal-dual:gpu -f Dockerfile-gpu .
```


