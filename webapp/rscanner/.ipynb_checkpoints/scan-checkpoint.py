from rscanner.model import RNN
from rscanner.util import lineToTensor, categoryFromOutput

import torch
import string
import pytesseract
import re
import cv2

from skimage.filters import threshold_local

import numpy as np

import imutils

all_letters = string.ascii_letters + " .,;'" + "äÄüÜöÖ"
n_letters = len(all_letters)

n_hidden = 128

rnn = RNN(n_letters, n_hidden, 3)

def order_coordinates(pts):
    rectangle = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    
    rectangle[0] = pts[np.argmin(s)]
    rectangle[2] = pts[np.argmax(s)]
    
    difference = np.diff(pts, axis=1)
    
    rectangle[1] = pts[np.argmin(difference)]
    rectangle[3] = pts[np.argmax(difference)]
    
    return rectangle


def point_transform(image, pts):
    rect = order_coordinates(pts)
    
    (upperLeft, upperRight, bottomRight, bottomLeft) = rect
    
    width1 = np.sqrt((bottomRight[0] - bottomLeft[0])**2 + (bottomRight[1] - bottomLeft[1])**2)
    width2 = np.sqrt((upperRight[0] - upperLeft[0])**2 + (upperRight[1] - upperLeft[1])**2)
    
    width = max(int(width1), int(width2))
    
    height1 = np.sqrt((upperRight[0] - bottomRight[0])**2 + (upperRight[1] - bottomRight[1])**2)
    height2 = np.sqrt((upperLeft[0] - bottomLeft[0])**2 + (upperLeft[1] - bottomLeft[1])**2)
    
    height = max(int(height1), int(height2))
    
    distance = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], dtype="float32")
    
    matrix = cv2.getPerspectiveTransform(rect, distance)
    
    warped_image = cv2.warpPerspective(image, matrix, (width, height))
    
    return warped_image
    
    

def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

def loadModel(path):
    rnn.load_state_dict(torch.load(path))

def scan(filePath, debug):
    gray_img = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)

    products = []

    kassenzettel = pytesseract.image_to_string(gray_img, lang='deu')


    kassenzettel = kassenzettel.replace("\n"," ")
    print(kassenzettel)
    # remove unwanted characters using regex
    regex = re.compile('[^a-zA-Z äÄüÜöÖ]')

    # crop the string using a keyword used by most receipts
    produkte = kassenzettel[:kassenzettel.upper().find("SUMME")]
    
    produkte = regex.sub('', produkte).upper()

    if debug:
        print(produkte)
    
    produkte = produkte.split()
    for i in range(len(produkte)):
        output = evaluate(lineToTensor(produkte[i]))
        category = categoryFromOutput(output)
        #print('%s (%d)' % (produkte[i], category))
        if category == 1:
            products.append(produkte[i])
        if category == 2 and i != 0:
            products[len(products)-1] += " " + produkte[i]

    return products