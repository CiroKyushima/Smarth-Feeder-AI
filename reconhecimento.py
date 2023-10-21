import cv2
import numpy as np

# Carregando o modelo YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Carregando classes
classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Carregando a imagem
image = cv2.imread("exemplos/cat_3.jpg")

# Pré-processando a imagem
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

# Passando a imagem para o modelo
net.setInput(blob)
outs = net.forward(net.getUnconnectedOutLayersNames())

# Lista para armazenar as caixas delimitadoras, confianças e classes
class_ids = []
confidences = []
boxes = []

# Processando as saídas do modelo
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Coordenadas da caixa delimitadora
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            w = int(detection[2] * image.shape[1])
            h = int(detection[3] * image.shape[0])

            # Pontos de canto da caixa delimitadora
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Aplicando a supressão não máxima para evitar caixas delimitadoras sobrepostas
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Desenhando caixas delimitadoras nas classes detectadas
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (242, 12, 12)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 5), font, 1, color, 1)

# Exibindo a imagem resultante
cv2.imshow("Detected Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()