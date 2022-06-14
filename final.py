#!/usr/local/bin/python
# coding: utf-8

################################################################################
# @remind Minhas importações
################################################################################
import hardware_monitor # Aplicação do hardware

################################################################################
# @remind Importações para criar aplicação com thread, ideal para raspberry pi
################################################################################
import threading
from multiprocessing import Queue
import threading
import sys
import time

# Inicia o seriço de monitoramento do hardware
lock = threading.Lock()  # Trava
app_queue = Queue() # Fila de envio de eventos
thread_monitor = threading.Thread(group=None, args=(
    lock, app_queue), target=hardware_monitor.hw_app_main, daemon=True)

###############################################################################
# @remind Configurações da Aplicação WEB/ REST API
###############################################################################
# Importações necessárias
import socket
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, redirect, jsonify, url_for, flash, Response, send_file, abort

app = Flask(__name__, static_url_path='/static', static_folder='./static')
app.config['JSON_AS_ASCII'] = False  # corrige os caracteres especiais
app.secret_key = "Uma chave qualquer"

###############################################################################
# @remind Rotas da aplicação
###############################################################################
@app.route('/')
@app.route('/home')
def index():
    # return render_template('index.html', status = thread_monitor.is_alive(), teste = entrada_liberacao)
    return render_template('index.html')

###############################################################################
# Leitura de frame de vídeo - @audit Pendente testar com mais de um acesso
###############################################################################
def gen():
    while True:
        frame = hardware_monitor.hw_api.get_frame_jpeg(lock)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

###############################################################################
# Leitura de frame de vídeo ja tratado - @audit Pendente testar com mais de um acesso
###############################################################################
def gen2():
    while True:
      # hardware_monitor.hw_api.testImage(lock)
      frame = hardware_monitor.hw_api.get_frame_tratado(lock)
      yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed_tratado')
def video_feed_tratado():
    return Response(gen2(), mimetype='multipart/x-mixed-replace; boundary=frame')
  
# Pega o IP da máquina. 
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception as e:
        print("erro: " + str(e))
        ip = "127.0.0.1"
    finally:
        s.close()
        return str(ip)
      
      
###############################################################################
# Iniciando a aplicação 
###############################################################################
if __name__ == "__main__":
  # Iniciando o serviço de monitoramento de hardware 
  if thread_monitor.is_alive() is False:
    thread_monitor.start()
  
  # Serviço flask
  app.run(host=get_ip(), port=80, debug=False)
  
  # Finalizando o serviço ded monitoramento de hardware 
  if (thread_monitor.is_alive() == True):
    app_queue.put_nowait('exit')
  
  print('Finalizando monitoramento.')
  print("FIM!")
  sys.stdout.flush()
  sys.exit()

# Liberar a camera
