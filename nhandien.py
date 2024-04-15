import cv2
import face_recognition

# step 1 :
imgAlon = face_recognition.load_image_file("image/anhmau1.jpg")
imgAlon = cv2.cvtColor(imgAlon,cv2.COLOR_BGR2RGB)
imgCheck = face_recognition.load_image_file("image/anhcheck.jpg")
imgCheck = cv2.cvtColor(imgCheck,cv2.COLOR_BGR2RGB)



# Xác đinh vị trí khuôn mặt cần nhận dạng
faceloc = face_recognition.face_locations(imgAlon)[0]
print(faceloc) #(y1,x2,y2,x1)
encodeElon = face_recognition.face_encodings(imgAlon)[0] # mã hóa ảnh
cv2.rectangle(imgAlon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceCheck = face_recognition.face_locations(imgCheck)[0]
encodeCheck = face_recognition.face_encodings(imgCheck)[0] # mã hóa ảnh
cv2.rectangle(imgCheck,(faceCheck[3],faceCheck[0]),(faceCheck[1],faceCheck[2]),(255,0,255),2)

# nó sẽ so sánh hình ảnh mã hóa với các điểm trên khuôn mặt xem có khớp o
results = face_recognition.compare_faces([encodeElon],encodeCheck)
print(results) # Kết quả True

# tuy nhiên khi có nhiều hình ảnh thì chúng ta cần phải biết
# khoảng cách (sai số ) giữa các bức ảnh là bao nhiêu?
faceDis = face_recognition.face_distance([encodeElon],encodeCheck)
print(results,faceDis)
#cv2.putText(imgAlon,f"{results}{(round(faceDis[0],2))}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.putText(imgCheck,f"{results}{(round(faceDis[0],2))}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv2.imshow("Alon",imgAlon)  # view thử ảnh để kiểm tra
cv2.imshow("AlonCheck",imgCheck) # view thử ảnh
cv2.waitKey()
cv2.destroyAllWindows()  # thoát tất cả các cửa sổ
