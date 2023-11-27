
from utils import launch_tensor_board

import threading
import time

if __name__ == '__main__':

    logdir = r'H:\By-FL\logs\2022-06-21\00.05.20'

    tb_port = 6004
    tb_host = "127.0.0.1"
    tb_thread = threading.Thread(
        target=launch_tensor_board,
        args=([logdir, tb_port, tb_host])
    ).start()
    print(logdir)
    time.sleep(3.0)

