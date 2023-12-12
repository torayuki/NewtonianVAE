docker run -d --gpus all -it --shm-size 32G --privileged -p 8000:8888 -p 6000:6006 -v ~/ws/sharespace:/root/sharespace --rm --name e-toramatsu-nvae e-toramatsu-nvae
