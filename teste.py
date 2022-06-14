# import os, fnmatch
# def listFiles(path, pattern):
#   listOfFiles = os.listdir(path)
#   array_list = []
#   for entry in listOfFiles:
#       if fnmatch.fnmatch(entry, pattern):
#         array_list.append(entry)
#   return array_list
# listFiles(PATH_QRCODES, EXT_QRCODE)
# listFiles(PATH_MODELS, EXT_MODELS)

# @remind read QRCode message
# detector = cv2.QRCodeDetector()
# data, bbox, straight_qrcode = detector.detectAndDecode(frame)
# if bbox is not None:
#   print("Information in the QRCode:", data)

################################################################################
# importações 
################################################################################
import cv2 
from src.objloader_simple import  *
import numpy as np
import math

################################################################################
# Lista de QR_Codes, aqui seria uma consulta no banco de dados
################################################################################
PATH_QRCODES = 'qrcode/'
EXT_QRCODE = '*.png'
PATH_MODELS = 'models/'
EXT_MODELS = '*.obj'
MIN_MATCHES = 120
CAMERA_PARAMETERS = np.array([[400, 0, 320], [0, 400, 240], [0, 0, 1]]) # Camera parameters

banco_de_dados = [
    {'id': 1, 'link': 'satc.edu.br', 'src': PATH_QRCODES +
        'qr_code_1_satc.png', 'model': PATH_MODELS + 'chair.obj', 'proportion_3D' : 30, 'color' : (255, 100, 0)}
    # ,
    # {'id': 2, 'link': 'OutroQRcode.com', 'src': PATH_QRCODES +
    #     'qr_code_2_satc.png', 'model': PATH_MODELS + 'fox.obj', 'proportion_3D' : 1}
    # ,
    # {'id': 3, 'link': 'teste.com', 'src': PATH_QRCODES +
    #     'qr_code_3_satc.png', 'model': PATH_MODELS + 'house.obj', 'proportion_3D' : 1}
]

# Inicia ORB detector
orb = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=10, fastThreshold=20, scaleFactor=1.5, WTA_K=4,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=500)
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
# NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2
################################################################################
# le os QR Codes
################################################################################
for item in banco_de_dados:
  item['img'] = cv2.imread(item['src'], 0)
  referenceImagePts, referenceImageDsc = orb.detectAndCompute(item['img'], None)
  item['referenceImagePts'] = referenceImagePts
  item['referenceImageDsc'] = referenceImageDsc
  item['obj'] = OBJ(item['model'], swapyz=True)

################################################################################
# projection_matrix
################################################################################
def projection_matrix(camera_parameters, homography):
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    
    return np.dot(camera_parameters, projection)

################################################################################
# project cube or model
################################################################################
def render(img, obj, projection, model, proportion_3D = 10, color=(80, 27, 211)):
    vertices = obj.vertices
    scale_matrix = np.eye(3) * proportion_3D
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)

        cv2.fillConvexPoly(img, imgpts, color)
    return img

def detect(sourceImage, banco = banco_de_dados):
  sourceImagePts, sourceImageDsc = orb.detectAndCompute(sourceImage, None)
  
  image_frame = sourceImage.copy()
  
  for item in banco:
    try:
      matches = bf.match(item['referenceImageDsc'], sourceImageDsc)
      # Sort them in the order of their distance
      matches = sorted(matches, key=lambda x: x.distance)
      # Apply the homography transformation if we have enough good matches
      print('matches', len(matches))
      
      if len(matches) > MIN_MATCHES:
          # Get the good key points positions
          sourcePoints = np.float32([item['referenceImagePts'][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
          destinationPoints = np.float32([sourceImagePts[m.trainIdx].pt for m in matches ]).reshape(-1, 1, 2)



          # Obtain the homography matrix
          homography, mask = cv2.findHomography(sourcePoints, destinationPoints, cv2.RANSAC, 5.0)
          matchesMask = mask.ravel().tolist()

          # Apply the perspective transformation to the source image corners
          h, w = item['img'].shape
          corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
          transformedCorners = cv2.perspectiveTransform(corners, homography)

          # Draw a polygon on the second image joining the transformed corners
          frame = cv2.polylines(sourceImage, [np.int32(transformedCorners)], True, 255, 3, cv2.LINE_AA)
          
          # obtain 3D projection matrix from homography matrix and camera parameters
          projection = projection_matrix(CAMERA_PARAMETERS, homography)  

          image_frame = render(frame, item['obj'], projection, item['img'], item['proportion_3D'], item['color'])
    except Exception as e:
      print("erro: " + str(e))

  return image_frame


# @audit Teste

# define a video capture object
vid = cv2.VideoCapture('https://192.168.1.198:8080/video')
      
while(True):
    ret, frame = vid.read()
    img_detect = detect(frame)
    
    cv2.imshow('img_detect', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()






    