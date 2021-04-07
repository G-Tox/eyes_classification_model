import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
import tensorflow as tf
import keras

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

#keras.backend.tensorflow_backend.set_session(get_session())	
#tensorflow.python.keras.api._v1.keras.backend

frame_id = 0
ran_num = np.random.randint(0,4)
cap = cv2.VideoCapture(0)
salida = cv2.VideoWriter('videoSalida.avi',cv2.VideoWriter_fourcc(*'XVID'),5.0,(640,480))
i = 1
while i == 1:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    if not ret:
        break

    elif ret == True:
	    detector = MTCNN()
	    recog=detector.detect_faces(img)
	    print(recog)

	    x_box = 200
	    y_box = 140
	    w_box = 250
	    h_box = 300

	    print(recog)

	    img = cv2.rectangle(img, (x_box,y_box), (x_box+w_box, y_box+h_box), (0,0,200), 3)
	    if recog:
	        box=recog[0]['box']
	        x=box[0]
	        y=box[1]
	        w=box[2]
	        h=box[3]
	        nose = recog[0]['keypoints']['nose']
	        img = cv2.circle(img,(nose[0],nose[1]),10,(255,0,255),3)
	        img = cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 3)
	        if (x > x_box) and (y > y_box) and ((x+w)<(x_box+w_box)) and ((y+h)<(y_box+h_box)):
	        	img = cv2.rectangle(img, (x_box,y_box), (x_box+w_box, y_box+h_box), (0,200,0), 3)
	        	box_nose = [[230,290,30,30],[230,340,30,30],[380,290,30,30],[380,340,30,30]]
	        	x_box_nose = box_nose[ran_num][0]
	        	y_box_nose = box_nose[ran_num][1]
	        	w_box_nose = box_nose[ran_num][2]
	        	h_box_nose = box_nose[ran_num][3]

	        	img = cv2.rectangle(img, (x_box_nose,y_box_nose), (x_box_nose + w_box_nose, y_box_nose + h_box_nose), (150,200,0), 3)

	        	if nose[0] >  x_box_nose and nose [1] > y_box_nose and nose[0] <  (x_box_nose + w_box_nose) and nose [1] < (y_box_nose + h_box_nose):
	        		texto = 'correcto'
	        		cv2.putText(img, texto, (10,30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
	        		i = 0
	        		#break
    
    cv2.imshow("Image", img)
    salida.write(img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
    			break 
    frame_id += 1
    imgHeight, imgWidth, channels = img.shape 
    print(imgHeight,imgWidth)

cap.release()
salida.release()
cv2.destroyAllWindows()