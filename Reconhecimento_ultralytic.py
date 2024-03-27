import cv2
import numpy as np
from ultralytics import YOLO

# Importação das funções e objectos necessários da biblioteca ultralítica
from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors

from collections import defaultdict  # Importando defaultdict do módulo collections

track_history = defaultdict(lambda: []) # Criando um defaultdict para armazenar o histórico de rastreamento dos objetos detectados

model = YOLO("yolov8n.pt") # Inicializando o modelo de detecção de objetos YOLO com pesos pré-treinados
names = model.model.names  # Carregando os nomes das classes do modelo YOLO

#video_path = "Reconhecimento_ultralytics\Teste.mp4" # Caminho para o arquivo de vídeo de entrada
#cap = cv2.VideoCapture(video_path)# Abrindo o arquivo de vídeo
cap = cv2.VideoCapture(0)#usando a webcam pra capturar imagem
assert cap.isOpened(), "Error reading video file" # Assertiva para verificar se o arquivo de vídeo foi aberto com sucesso

# Recuperando as propriedades do vídeo: largura, altura e frames por segundo (fps)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Criando um objeto VideoWriter para escrever o vídeo processado com as detecções
result = cv2.VideoWriter("Reconhecimenot_ultralytic.avi",
                       cv2.VideoWriter_fourcc(*'mp4v'),# Codec usado para escrever o vídeo
                       fps,
                       (w, h))# Tamanho do quadro

while cap.isOpened(): # Looping através de cada quadro do vídeo
    success, frame = cap.read() # Lendo um quadro do vídeo
    if success: # Se o quadro for lido com sucesso
        results = model.track(frame, persist=True, verbose=False)  # Detectando e rastreando objetos no quadro usando o modelo YOLO
        boxes = results[0].boxes.xyxy.cpu() # Extraindo caixas delimitadoras dos objetos detectados

        if results[0].boxes.id is not None:  # Se os IDs dos objetos estiverem disponíveis

            # Extraindo classe, ID de rastreamento e pontuação de confiança dos objetos detectados
            clss = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.float().cpu().tolist()

            # Inicializando o objeto Annotator para desenhar caixas delimitadoras e rótulos
            annotator = Annotator(frame, line_width=2)

            # Iterando através de cada objeto detectado
            for box, cls, track_id in zip(boxes, clss, track_ids):
                # Desenhando caixa delimitadora e rótulo para o objeto detectado
                annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

                # Armazenando o histórico de rastreamento do objeto detectado
                track = track_history[track_id]
                track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                if len(track) > 30: # Removendo a posição mais antiga se o histórico exceder um certo comprimento
                    track.pop(0)

                # Desenhando trilhas do objeto no quadro
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1) # Desenhando círculo na posição atual
                cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2) # Desenhando polilinha conectando posições anteriores
        
        cv2.imshow("detections", frame)
        
        result.write(frame) # Escrevendo o quadro anotado no arquivo de vídeo de saída
        if cv2.waitKey(1) & 0xFF == ord("q"): # Verificando se a tecla 'q' foi pressionada para sair do loop
            break
    else:
        break # Interrompendo o loop se o final do vídeo for alcançado

# Liberando os objetos de escrita e captura de vídeo
result.release()
cap.release()
# Fechando todas as janelas do OpenCV
cv2.destroyAllWindows()