import os
import sys
from time import sleep
import cv2 
import numpy as np
import math
from src.objloader_simple import  *

# Iniciando a camera do hardware
import hardware_api
CAMERA_ADDR = 'http://192.168.1.198:8080/video'
# Instanciando a camera do rapiberry ou o cv2
hw_api = hardware_api.hardwareAPI(hardware_api.camerapi_ok, endereco=CAMERA_ADDR, width=800, height=600)

## Receber notificações da aplicação principal
def verifica_notificacao(lock, app_queue):
    while not app_queue.empty():
      data = app_queue.get()
      print(data)

def hw_app_main(lock, app_queue):
  try:
    print('init task_monitor')
    # Iniciar a camera antes do setup
    hw_api.update_frame(lock)
    while True:
      # Verifica a operação
      verifica_notificacao(lock, app_queue)
      hw_api.testImage(lock)
      # print('run')
      # sleep(0.025)
  finally:
    print('\r\n\r\nO monitoramento de hardware foi finalizado\r\n')
    sys.stdout.flush()  