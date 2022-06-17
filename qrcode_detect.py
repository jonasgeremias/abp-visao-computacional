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
MIN_MATCHES = 35
MIN_AREA = 5000
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


def detect(sourceImage, banco=banco_de_dados, camera_parameters=CAMERA_PARAMETERS):
    sourceImagePts, sourceImageDsc = orb.detectAndCompute(sourceImage, None)
    image_frame = sourceImage.copy()
    array_info_detect = []
    for item in banco:
        try:
            matches = detect_bf(item['referenceImageDsc'], sourceImageDsc)
            # matches = detect_flann(item['referenceImageDsc'], sourceImageDsc)
            # Apply the homography transformation if we have enough good matches
            # print('matches', len(matches))

            # if len(matches) >= MIN_MATCHES:
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
            # array_info_detect.append({'id': item['id'], 'matches': len(matches),'area': area, 'approx': len(approx), 'shape_factor': shape_factor})
            
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
    
    # print(array_info_detect)
  
    return image_frame

# @audit Teste
# # define a video capture object
# vid = cv2.VideoCapture('http://192.168.1.198:8080/video')
# while(True):
#     ret, frame = vid.read()
#     img_detect = detect(frame)
#     cv2.imshow('img_detect', img_detect)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# vid.release()
# cv2.destroyAllWindows()
