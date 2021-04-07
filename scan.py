from mtcnn.mtcnn import MTCNN
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('models/model.h5')
frame_id = 0
cap = cv2.VideoCapture(0)
while (True):
    ret, img = cap.read()
    if not ret:
        break

    elif ret == True:
	    detector = MTCNN()
	    recog=detector.detect_faces(img)
	    print(recog)

	    if recog:
	        box=recog[0]['box']
	        x=box[0]
	        y=box[1]
	        w=box[2]
	        h=box[3]

	        crop_face = img[y-50:(y)+(h+50),(x-50):(x)+(w+50)]
	        crop_face = cv2.resize(crop_face, (830,900))
	        #cv2.imwrite("dataset/img_test/resize.jpg", crop_face)
	        img = cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 10)

	        detector8 = MTCNN()
	        recog_eyes = detector8.detect_faces(crop_face)

	        if recog_eyes:

	            ojo_iz = recog_eyes[0]['keypoints']['left_eye']
	            ojo_de = recog_eyes[0]['keypoints']['right_eye']

	            ojoiz_x=ojo_iz[0]
	            ojoiz_y=ojo_iz[1]

	            ojosd_x=ojo_de[0]
	            ojosd_y=ojo_de[1]

	            ojoiz_new_x = ojoiz_x - 120
	            ojoiz_new_y = ojoiz_y - 90
	            ojo_w = 200
	            ojo_h = 150

	            ojode_new_x = ojosd_x - 110
	            ojode_new_y = ojosd_y - 90

	            #ojoiz_new_x = ojoiz_x - 230
	            #ojoiz_new_y = ojoiz_y - 150
	            #ojo_w = 450
	            #ojo_h = 300

	            #ojode_new_x = ojosd_x - 200
	            #ojode_new_y = ojosd_y - 150


	            #img = cv2.rectangle(img, (ojoiz_new_x,ojoiz_new_y), (ojoiz_new_x+ojo_w, ojoiz_new_y+ojo_h), (100,100,100), 2)
	            #img = cv2.rectangle(img, (ojode_new_x,ojode_new_y), (ojode_new_x+ojo_w, ojode_new_y+ojo_h), (25,100,45), 2)
	            #img = cv2.circle(img,(ojoiz_x,ojoiz_y),10,(255,0,255),3)
	            #img = cv2.circle(img,(ojosd_x,ojosd_y),20,(255,0,255),3)

	            #cv2.imwrite("dataset/img_test/cerrar-ojos-12.jpg", img)

	            crop_ojo_der = crop_face[ojode_new_y:ojode_new_y+ojo_h, ojode_new_x:ojode_new_x+ojo_w]
	            output = cv2.resize(crop_ojo_der, (200,200))
	            #cv2.imwrite("dataset/img_test/resize_ojo.jpg", crop_ojo_der)
	            x=image.img_to_array(output)
	            x=np.expand_dims(x, axis=0)
	            images = np.vstack([x])
	            classes = None
	            classes = model.predict(images, batch_size=10)

	            if classes >=0.5:
	                texto = 'Ojo Izquierdo abierto'
	                cv2.putText(img, texto, (10,10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
	                cv2.putText(img, texto, (10,10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

	            elif classes < 0.5:
	                texto = 'Ojo Izquierdo cerrado'
	                cv2.putText(img, texto, (10, 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
	                cv2.putText(img, texto, (10, 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

	            crop_ojo_iz = crop_face[ojoiz_new_y:ojoiz_new_y+ojo_h, ojoiz_new_x:ojoiz_new_x+ojo_w]
	            output_1 = cv2.resize(crop_ojo_iz, (200,200)) 
	            #cv2.imwrite("dataset/img_test/resize_1.jpg", crop_ojo_iz)
	            x_1=image.img_to_array(output_1)
	            x_1=np.expand_dims(x_1, axis=0)
	            images_1 = np.vstack([x_1])
	            classes_1 = None
	            classes_1 = model.predict(images_1, batch_size=10)

	            if classes_1 >=0.5:
	                texto = 'Ojo derecho abierto'
	                cv2.putText(img, texto, (30,30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
	                cv2.putText(img, texto, (30,30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

	            elif classes_1 < 0.5:
	                texto = 'Ojo derecho cerrado'
	                cv2.putText(img, texto, (30,30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
	                cv2.putText(img, texto, (30,30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

	        else:
	        	print(' no se reconocio los ojos')   		    

    else: break

    #cv2.imwrite("dataset/scan/"+str(frame_id)+".jpg", img)    
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
    			break 
    frame_id += 1

cap.release()
cv2.destroyAllWindows()