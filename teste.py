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

################################################################################
# importações
################################################################################
import time
import pyzbar.pyzbar as pyzbar
from pickle import NONE
import cv2
from src.objloader_simple import *
import numpy as np
import math

################################################################################
# Lista de QR_Codes, aqui seria uma consulta no banco de dados
################################################################################
PATH_QRCODES = 'qrcode/'
EXT_QRCODE = '*.png'
PATH_MODELS = 'models/'
EXT_MODELS = '*.obj'
MIN_MATCHES = 80
MIN_AREA = 1000
SHAPE_FACTOR_MIN = 200
SHAPE_FACTOR_MAX = 1000

CAMERA_PARAMETERS = np.array(
    [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])  # Camera parameters

banco_de_dados = [
    # {'id': 1, 'link': 'satc.edu.br', 'src': PATH_QRCODES +
    # 'qr_code_1_satc.png', 'model': PATH_MODELS + 'chair.obj', 'proportion_3D': 20, 'color': (255, 100, 0)},
    {'id': 1, 'link': 'satc.edu.br', 'src': PATH_QRCODES +
        'qr_code_1_satc.png', 'model': PATH_MODELS + 'satc/SATC.obj', 'proportion_3D': 300, 'color': (255, 100, 0)},
    {'id': 2, 'link': 'OutroQRcode.com', 'src': PATH_QRCODES +
        'qr_code_2_satc.png', 'model': PATH_MODELS + 'fox.obj', 'proportion_3D': 8, 'color': (100, 100, 0)},
    {'id': 3, 'link': 'teste.com', 'src': PATH_QRCODES +
        'qr_code_3_satc.png', 'model': PATH_MODELS + 'chair.obj', 'proportion_3D': 20, 'color': (0, 100, 255)}
]

# Inicia ORB detector
orb = cv2.ORB_create(WTA_K=3, scoreType=cv2.ORB_FAST_SCORE,
                     firstLevel=0, nfeatures=500)

###########################
# Test flann
###########################
index_params = dict(algorithm=6, table_number=6,
                    key_size=2, multi_probe_level=1)
search_params = {}
flann = cv2.FlannBasedMatcher(index_params, search_params)


def detect_flann(referenceImageDsc, sourceImageDsc):
    matches = flann.knnMatch(referenceImageDsc, sourceImageDsc, k=2)
    # As per Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches


###########################
# Test BFMatcher
###########################
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2


def detect_bf(referenceImageDsc, sourceImageDsc):
    matches = bf.match(referenceImageDsc, sourceImageDsc)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


################################################################################
# le os QR Codes
################################################################################
for item in banco_de_dados:
    item['img'] = cv2.imread(item['src'], 0)
    referenceImagePts, referenceImageDsc = orb.detectAndCompute(
        item['img'], None)
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
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d /
                   np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d /
                   np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)

    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T

    return np.dot(camera_parameters, projection)

################################################################################
# project cube or model
################################################################################


def render(img, obj, projection, model, proportion_3D=10, color=(80, 27, 211)):
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


def calcula_scores(banco, sourceImagePts, sourceImageDsc):
    array_detect = []
    for item in banco:
        try:
            matches = detect_bf(item['referenceImageDsc'], sourceImageDsc)

            # Get the good key points positions
            sourcePoints = np.float32(
                [item['referenceImagePts'][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            destinationPoints = np.float32(
                [sourceImagePts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Obtain the homography matrix
            homography, mask = cv2.findHomography(
                sourcePoints, destinationPoints, cv2.RANSAC, 5.0)

            # Apply the perspective transformation to the source image corners
            h, w = item['img'].shape
            corners = np.float32(
                [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

            # Draw a polygon on the second image joining the transformed corners
            transformedCorners = cv2.perspectiveTransform(
                corners, homography)

            ################################################################
            # Evita detecções de QR code falsos ou muito pequenos.
            ################################################################
            peri = 0.01 * cv2.arcLength(transformedCorners, True)
            approx = cv2.approxPolyDP(
                transformedCorners, peri, True)  # 4 pontos
            area = cv2.contourArea(approx)  # Area muito pequena é ignorada
            shape_factor = area / (peri * peri)
            approx_len = len(approx)

            # pontos tem formato de quadrado, retangulo
            if (approx_len == 4):
                # print({'matches' : len(matches), 'area': area, 'approx_len': approx_len, 'approx': approx, 'shape': shape_factor, 'homography': homography})
                array_detect.append({'item': item, 'matches': len(
                    matches), 'area': area, 'approx_len': approx_len, 'approx': approx, 'shape': shape_factor, 'homography': homography})
        except Exception as e:
            print("erro: " + str(e))
    return array_detect


def filter_better_score(array_detect):
    better_item = {}
    #  Nenhuma detecção
    if (len(array_detect) == 0):
        return NONE
    #  Só uma detecção
    better_item = array_detect.pop(0)
    if (len(array_detect) == 0):
        return better_item
    # Procura o melhor resultado
    for item in array_detect:
        if (item['matches'] > better_item['matches']):
            # if (item['shape'] >= SHAPE_FACTOR_MIN and item['shape'] < SHAPE_FACTOR_MAX):
            if (item['area'] >= MIN_AREA and item['area'] >= better_item['area']):
                better_item = item
    return better_item


def detect(sourceImage, banco=banco_de_dados, camera_parameters=CAMERA_PARAMETERS):
    sourceImagePts, sourceImageDsc = orb.detectAndCompute(sourceImage, None)
    image_frame = sourceImage.copy()

    for item in banco:
        try:
            matches = detect_bf(item['referenceImageDsc'], sourceImageDsc)
            # matches = detect_flann(item['referenceImageDsc'], sourceImageDsc)
            # Apply the homography transformation if we have enough good matches
            # print('matches', len(matches))
            # Get the good key points positions
            sourcePoints = np.float32(
                [item['referenceImagePts'][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            destinationPoints = np.float32(
                [sourceImagePts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            # Obtain the homography matrix
            homography, mask = cv2.findHomography(
                sourcePoints, destinationPoints, cv2.RANSAC, 5.0)
            # matchesMask = mask.ravel().tolist()
            # Apply the perspective transformation to the source image corners
            h, w = item['img'].shape
            corners = np.float32(
                [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            # Draw a polygon on the second image joining the transformed corners
            transformedCorners = cv2.perspectiveTransform(
                corners, homography)
            ################################################################
            # Evita detecções de QR code falsos ou muito pequenos.
            ################################################################
            peri = 0.01 * cv2.arcLength(transformedCorners, True)
            approx = cv2.approxPolyDP(
                transformedCorners, peri, True)  # 4 pontos
            area = cv2.contourArea(approx)  # Area muito pequena é ignorada
            shape_factor = area / (peri * peri)

            print(len(matches), area, shape_factor, len(approx))
            if area >= MIN_AREA and len(approx) == 4 and shape_factor >= 500 and shape_factor < 1000:
                pts = [np.int32(approx)]
                image_frame = cv2.polylines(
                    sourceImage, pts, True, item['color'], 3, cv2.LINE_4)

                # image_frame = frame
                # # obtain 3D projection matrix from homography matrix and camera parameters
                projection = projection_matrix(
                    camera_parameters, homography)
                image_frame = render(
                    image_frame, item['obj'], projection, item['img'], item['proportion_3D'], item['color'])
        except Exception as e:
            print("erro: " + str(e))

    return image_frame


################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

def detect_by_text(decodedObject, banco=banco_de_dados):
    for item in banco:
      txtqr = decodedObject.data
      detect = txtqr.find(item['link'].encode())
      if detect > -1:
        return item

    return None

# get the webcam:
cap = cv2.VideoCapture('https://192.168.1.198:4747/video')
cap.set(3, 640)
cap.set(4, 480)
time.sleep(2)
font = cv2.FONT_HERSHEY_SIMPLEX

def drawframe(frame, hull, decodedObject):
  # Number of points in the convex hull
  n = len(hull)
  # Draw the convext hull
  for j in range(0, n):
      cv2.line(frame, hull[j], hull[(j+1) % n], (255, 0, 0), 3)
  x = decodedObject.rect.left
  y = decodedObject.rect.top
  print(x, y)
  print('Type : ', decodedObject.type)
  print('Data : ', decodedObject.data, '\n')
  barCode = str(decodedObject.data)
  cv2.putText(frame, barCode, (x, y), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
  
  return frame

while(cap.isOpened()):
    ret, frame = cap.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    decodedObjects = pyzbar.decode(img_gray)

    for decodedObject in decodedObjects:
        points = decodedObject.polygon
        # If the points do not form a quad, find convex hull
        if len(points) > 4:
            hull = cv2.convexHull( np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else: hull = points

        frame = drawframe(frame, hull, decodedObject)
        item = detect_by_text(decodedObject)
        
        if item != None:
          print(item['id'])
          sourceImagePts, sourceImageDsc = orb.detectAndCompute(frame, None)
          matches = detect_bf(item['referenceImageDsc'], sourceImageDsc)
          sourcePoints = np.float32( [item['referenceImagePts'][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
          destinationPoints = np.float32([sourceImagePts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

          pontos = np.float32(hull)
          pontosinv = np.float32(hull).reshape(-1, 1, 2)
          # # Obtain the homography matrix
      
          homography, mask = cv2.findHomography(sourcePoints, destinationPoints, cv2.RANSAC, 5.0)   
          
          projection = projection_matrix(CAMERA_PARAMETERS, homography)
          
          
          print('pontos', pontos)
          print('pontosinv', pontosinv)
          print('destinationPoints', destinationPoints)
          
          # image_frame = render(frame, item['obj'], hull, item['img'], item['proportion_3D'], item['color'])
        
    # Display the resulting frame
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'): 
      break
    elif key & 0xFF == ord('s'):  # wait for 's' key to save
        cv2.imwrite('Capture.png', frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
