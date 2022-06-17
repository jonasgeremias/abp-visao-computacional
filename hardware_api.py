import cv2
import qrcode_detect
import numpy as np

IMG_MIRROR = False
TEST_IMG = False

## Se for num raspberry Pi, inicia a camera
camerapi_ok = 0
try:
    import picamera
    from picamera.array import PiRGBArray
    print('hardware_api com picamera')
    camerapi_ok = 1
except Exception as e:
    print("erro: " + str(e))

class hardwareAPI(object):
  def __init__(self, habilita_picamera=0, endereco=0, width=1280, height=720):
    try:
      self.habilita_picamera = habilita_picamera
      self.frame_tratado = []
      self.width = width
      self.height = height
      self.json_config = {}
      if (self.habilita_picamera):
          self.camera = picamera.PiCamera()
          self.camera.resolution = (height, width)
          print("picamera")
      else:
          if TEST_IMG == 0:
            self.camera = cv2.VideoCapture(endereco)
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
            self.camera.set(3, self.width)
            self.camera.set(4, self.height)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
          print("cv2 link")
    finally:
        self.frame_tratado = self.get_frame()
        print("Camera está iniciada")
  def __del__(self):
    if (self.habilita_picamera):
        self.camera.close()
    else:
        self.camera.release()
  
  def get_frame(self):
    frame = []
    if (self.habilita_picamera):
        self.rawCapture = PiRGBArray(self.camera)
        self.camera.capture(self.rawCapture, format="bgr")
        frame = self.rawCapture.array
        #ret, jpeg = cv2.imencode('.jpg', frame)
    else:
      if TEST_IMG == 0:
        # ret, frame = self.camera.read()
        ret, frame = self.camera.read()
      else:
        frame = cv2.imread('images\Exemplo_QR_code_1.jpeg')
        
        # frame = cv2.resize(frame, (400,600), interpolation = cv2.INTER_AREA)
    frame = cv2.resize(frame, (self.height, self.width), interpolation = cv2.INTER_AREA)
    # Escolha qual o retorno
    # return jpeg.tobytes()
    return frame
    
    #atualiza o frame e converte para JPG
  def get_frame_jpeg(self, lock):
    lock.acquire()
    ret, jpeg = cv2.imencode('.jpeg', self.get_frame())
    frame = jpeg.tobytes()
    lock.release()
    return frame
  
  # Envia o frame com alterações
  def get_frame_tratado(self, lock):
    lock.acquire()
    img = cv2.resize(self.frame_tratado, (self.height, self.width), interpolation = cv2.INTER_AREA)
    ret, jpeg = cv2.imencode('.jpg', img)
    img = jpeg.tobytes()
    lock.release()
    return img

  # Atualiza o frame
  def update_frame(self, lock):
    lock.acquire()
    frame = self.get_frame()
    lock.release()
    return frame
  
  # Atualiza o frame tratado
  def update_frame_tratado(self, lock, image):
      lock.acquire()
      self.frame_tratado = image
      lock.release()

  def testImage(self, lock):
      img = self.update_frame(lock)        
      image_preview = img.copy()
      
      if IMG_MIRROR:
        image_preview = cv2.flip(image_preview, 1)
            
      # Busca o json com as configurações
      try:
        # CAMERA_PARAMETERS = np.array([[1000, 0, self.width], [0, 1000, self.height], [0, 0, 1]])  # Camera parameters
        image_preview = qrcode_detect.detect(image_preview) #, camera_parameters = CAMERA_PARAMETERS)
        # print('detect')

      except Exception as e:
        print("erro: " + str(e))

      self.update_frame_tratado(lock, image_preview)
      return True