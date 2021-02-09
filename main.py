import numpy as np
import cv2

# Definição das Variaveis
threshold = 0.5             # Limite minimo pra detecção valida
threshold_NMS = 0.4         # limite sobreposição
cont_car = 0                # Variavel para contar os carros
cont = False                # Condição pro contador ( Estado da ultima leitura)

video = cv2.VideoCapture('videoeditado.mp4') #Declarando a entrada de video

# Video de Saida
fourcc = cv2.VideoWriter_fourcc(*'XVID')
saida = cv2.VideoWriter('resultado.avi', fourcc, 20.0, (int(video.get(3)), (int(video.get(4)))))

# Yolo
labelspath = "yolobj.names"                 # Arquivo com os nomes dos  objetos treinados
labels = open(labelspath).read().strip().split()        #leitura do nomes
weightspath = "yolov3.weights"                      
configpath = "yolov3.cfg"
net = cv2.dnn.readNet(configpath, weightspath)

np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")    
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Função Blob
def blob_img(net, roi):
    blob = cv2.dnn.blobFromImage(roi, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    return net, roi, layerOutputs


# Função pra detecção
def deteccoes(detection, _threshold, boxes, confiancas, classes):
    scores = detection[5:]
    classeID = np.argmax(scores)
    confianca = scores[classeID]

    if confianca > _threshold:
        caixa = detection[0:4] * np.array([W, H, W, H])
        (centerX, centerY, width, height) = caixa.astype("int")

        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))

        boxes.append([x, y, int(width), int(height)])
        confiancas.append(float(confianca))
        classes.append(classeID)
    return boxes, confiancas, classes


# Detecção no video
def info(frame, i, boxes, labels):
    (x, y) = (boxes[i][0], boxes[i][1])
    (w, h) = (boxes[i][2], boxes[i][3])

    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 0, 0), 2)        # Desenhando retangulo das detecões

    if labels[classes[i]] == 'car':                                 # Verifica se o objeto detectado é um carro
        if globals()['cont'] is False:                              
            globals()['cont_car'] += 1  
            globals()['cont'] = True                                # Marca o contador como verdadeiro pra não somar a detecção do proximo frame

    return frame, x, y, w, h


while True:

    ret, frame = video.read()   # Leitura dos frames do video 

    txt1 = "Carros: {}".format(cont_car)
    cv2.putText(frame, txt1, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)         # Texto do contador de carros

    if ret is True:
        cv2. rectangle(frame, (680, 500), (880, 610), (128, 128, 128), 2)           #Desenho retangulo da area de interesse
        roi = frame[500:610, 680:880]                       # Dimensoes da area de interesse
        (H, W) = roi.shape[:2]                              # Atribundo as dimensoes as variaveis de altura e largura
                          
        net, roi, layerOutputs = blob_img(net, roi)         # Chamadno a funçao de blob
        boxes = []
        confiancas = []
        classes = []

        for output in layerOutputs:
            for detection in output:
                boxes, confiancas, classes = deteccoes(detection, threshold, boxes, confiancas, classes)    # Função pra detecção dos veiculos

        objs = cv2.dnn.NMSBoxes(boxes, confiancas, threshold, threshold_NMS)            # Retirar as caixar que se sobrepoem

        if len(objs) > 0:
            for i in objs.flatten():
                roi, x, y, w, h = info(frame, i, boxes, labels)     
        else:
            globals()['cont'] = False
        saida.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

 # Liberando os videos 
video.release()
saida.release()
cv2.destroyAllWindows()
