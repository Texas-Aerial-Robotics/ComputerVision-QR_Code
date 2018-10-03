from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2

import pygame
import pygame.camera

pygame.camera.init()
pygame.camera.list_cameras() #Camera detected or not
cam = pygame.camera.Camera("/dev/video0",(640,480))
cam.start()
img = cam.get_image()
pygame.image.save(img,"/home/faizfb2000/cv_tutorial/qrtest4.png")
# /home/faizfb2000/cv_tutorial qrtest4.png
 
def decode(im) : 
  # Find barcodes and QR codes
  decodedObjects = pyzbar.decode(im)
 
  # Print resultspython
  for obj in decodedObjects:
    print('Type : ', obj.type)
    print('Data : ', obj.data,'\n')
     
  return decodedObjects
 
 
# Display barcode and QR code location  
def display(im, decodedObjects):
 
  # Loop over all decoded objects
  for decodedObject in decodedObjects: 
    points = decodedObject.polygon
 
    # If the points do not form a quad, find convex hull
    if len(points) > 4 : 
      hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
      hull = list(map(tuple, np.squeeze(hull)))
    else : 
      hull = points;
     
    # Number of points in the convex hull
    n = len(hull) 
 
    # Draw the convext hull
    for j in range(0,n):
      cv2.line(im, hull[j], hull[ (j+1) % n], (255,0,0), 3)
 
  # Display results 
  cv2.imshow("Results", im);
  cv2.waitKey(0);
 
   
# Main 
if __name__ == '__main__':
 
  # Read image
  im = cv2.imread('qrtest4.png')
 
  decodedObjects = decode(im)
  display(im, decodedObjects)