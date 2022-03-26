# CPSC540-project
UBC CPSC 540 course project 

# How to run
1. Example notebook: DANN_train.ipynb

2. Can directly run python scripts (recommended):
    ```
    python3 train.py
    ```

# Package Dependency
pytorch, torchvision, opencv, sklearn, pickle, tensorboard, tqdm


# View Remote Tensorboard
1. Connect to server through SSH and map the port between the server and the host computer.
    ```
    ssh -L <host_port>:localhost:<remote_port> user@remote
    ```
    For example:
    ```
    ssh -L 6006:localhost:6006 user@remote
    ```
2. Map the port in the container with the server.
    ```
    docker run -it -p <remote_port>:<container_port> --volume=$HOME:/workspace --name=name_of_container --gpus device=number name_of_image
    ```
    Usually the container port should be 6006 (the defualt tensorboard port).
    For example:
    ```
    docker run -it -p 6006:6006 --volume=$HOME:/workspace --name=name_of_container --gpus device=number name_of_image
    ```
3. Run train.py, the SummaryWriter will automatically start a tensorboard and log in cfg['tensorboard_dir'].

4. Open another SSH connection in a new window, use the port in the first step.

5. Enter the same container, and enter the code folder. 
    ```
    docker exec -w /workspace -it name_of_container bash
    cd /workspace/your_folder
    ```
6. Find the dir we set to save the tensorboard log. Run tensorboard and allow port network mapping in terminal.
    ```
    tensorboard --logdir <cfg['tensorboard_dir']> --bind_all
    ```
6. In host computer, open a browser and view tensorboard in this address:
    ```
    http://localhost:<host_port>
    ```