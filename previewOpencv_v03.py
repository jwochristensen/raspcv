import AdapterBoard_v03 as AdapterBoard
import cv2 as cv
import numpy as np

Arducam_adapter_board = AdapterBoard.MultiAdapter()
if __name__ == "__main__":
    Arducam_adapter_board.init(320,240)
    Arducam_adapter_board.preview()

    #Arducam_adapter_board.preview_channel(0)
