from fastai import Path
import cv2
import numpy as np
from fastai.vision import Image, pil2tensor
import matplotlib.pyplot as plt

ROOT = Path('data/demonstration')
suffixes = ['_red.png', '_green.png', '_blue.png', '_yellow.png']
prompt = "Do you want to make predictions based on a folder or ID (1 or 2) ? "

def open_im_by_id(ID, ROOT=ROOT, sufx=suffixes):
    """
    ID : Base name for each sample
    ROOT : Folder containing all samples
    sufx : color suffixes for samples
    """
    imnames = [ROOT/ID/(ID + o) for o in sufx] # Generating file names for given image ID
    imgs = [cv2.imread(str(o), cv2.IMREAD_GRAYSCALE) for o in imnames]  # List of 4 channels of given file ID
    imgs = np.stack(imgs, 2)
    return Image(pil2tensor(imgs, np.float32).float())  # Creating a Fastai Image object from the image

def open_im_by_folder(FOLDER, sufx=suffixes):
    """
    FOLDER : Folder containing sample files
    sufx : color suffixes for samples
    """
    FOLDER = Path(FOLDER)
    fnames = sorted(FOLDER.ls())
    fnames = [fnames[2], fnames[1], fnames[0], fnames[3]] # Sorting according to RGBY from BGRY
    imgs = [cv2.imread(str(o), cv2.IMREAD_GRAYSCALE) for o in fnames]  # List of 4 channels of given file ID
    imgs = np.stack(imgs, 2)
    return Image(pil2tensor(imgs, np.float32).float())  # Creating a Fastai Image object from the image

def get_folder_by_ID(ROOT=ROOT):
    is_repeating = False
    is_ID_valid = False
    while not is_ID_valid:
        prompt = "Enter the ID of the samples : " if not is_repeating \
                 else "Please enter a valid ID : "
        try:
            ID = Path(input(prompt))
            folder = ROOT/ID
            if not folder.is_dir():
                raise FileNotFoundError
            else:
                is_ID_valid = True
        except FileNotFoundError:
            is_repeating = True
            is_ID_valid = False

    return folder

def get_folder():
    is_repeating = False
    is_folder_valid = False
    while not is_folder_valid:
        prompt = "Enter the location of the folder containing the samples : " if not is_repeating \
                 else "Please enter a valid folder : "
        try:
            folder = Path(input(prompt))
            if not folder.is_dir():
                raise FileNotFoundError
            else:
                is_folder_valid = True
        except FileNotFoundError:
            is_repeating = True
            is_folder_valid = False

    return folder

def get_user_intent(prompt: str = prompt) -> Image:
    FOLDER = '1'
    intent = input(prompt)
    if intent == 'exit':
        return False
    is_folder = True if intent == FOLDER else False
    if is_folder:
        folder = get_folder()
    else:
        folder = get_folder_by_ID()
    return open_im_by_folder(folder)

def get_np_image(img: Image) -> np.ndarray:
    return np.transpose(img.data.numpy(), (1,2,0)).astype(int)


def show_sample(img: Image):
    image = get_np_image(img)
    fig, ax = plt.subplots(1, 2, figsize=(20,10))
    ax[0].imshow(image[:,:, [0,3,2]])
    ax[1].imshow(image[:,:, 1], cmap='gray')
    ax[0].set_title('Cell')
    ax[1].set_title('Protein')
    fig.show()
