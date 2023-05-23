# 아이펠캠퍼스 온라인4기 피어코드리뷰[23.05.23]

- 코더 : 부석경
- 리뷰어 : 이성주

---------------------------------------------
## **PRT(PeerReviewTemplate)**

### **[O] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
1. 자기만의 카메라앱 기능 구현을 완수하였다.
   - 사진이 너무 커서 얼굴의 랜드마크를 제대로 찾지 못한 경우도 있지만, 스티커의 위치는 코 위치를 이용하여 잘 찾았습니다.

2. 스티커 이미지를 정확한 원본 위치에 반영하였다.	
   - 스티커를 턱 맨아래점과 코 위에점을 이용하여 각도를 계산하여 스티커를 회전시켜 스티커를 정확하게 위치 시켰습니다.

3. 카메라 스티커앱을 다양한 원본이미지에 적용했을 때의 문제점을 체계적으로 분석하였다.	얼굴각도, 이미지 밝기, 촬영거리 등 다양한 변수에 따른 영향도를 보고서에 체계적으로 분석하였다.
   - 얼굴 각도에 따라서 테스트를 진행하였고, 빛의 위치 촬영거리, 악세서리를 이용한 사진들을 시도 하며 분석하였습니다.
 
### **[o] 주석을 보고 작성자의 코드가 이해되었나요?**
``` python
def My_sticker_app(image,img_sticker,landmark_predictor,img_Transparency = 0.7, hog_pypamid = 1):
    img = image.copy()
    # face detector
    detector_hog = dlib.get_frontal_face_detector() # 기본 얼굴 감지기를 반환
    dlib_rects = detector_hog(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), hog_pypamid)   # (image, num of image pyramid)

    # landmark detector
    list_landmarks = []
    for dlib_rect in dlib_rects:
        points = landmark_predictor(img, dlib_rect) # 모든 landmark의 위치정보를 points 변수에 저장
        list_points = list(map(lambda p: (p.x, p.y), points.parts())) # 위 사진의 점들을 의미
        list_landmarks.append(list_points)  # list_landmarks에 랜드마크 리스트를 저장

    list_stickers_point = []
    for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
        w = h = (landmark[35][0] - landmark[31][0])*7 # 코의 좌우 크기를 비율로 스티커 이미지 사이즈 조절
        
        x = landmark[30][0] - w//2  # 이미지에서 코 부위의 x값
        y = (landmark[30][1]+landmark[33][1])//2 - w//2  # 이미지에서 코 부위의 y값
        
        # 고양이 수염의 각도 조정
        dy = landmark[8][1] - landmark[27][1]
        dx = landmark[8][0] - landmark[27][0]
        sticker_angle = math.atan2(dx, dy) * (180.0 / math.pi)
        
        # 회전 행렬 생성후 회전
        img_center = tuple((img_sticker.shape[0]/2,img_sticker.shape[1]/2))
        rot_mat = cv2.getRotationMatrix2D(img_center, sticker_angle, 1.0) 
        img_sticker = cv2.warpAffine(img_sticker, rot_mat, img_sticker.shape[1::-1], flags=cv2.INTER_LINEAR,borderValue=(255,255,255))
        
        # 스티커가 원본사진을 넘어가면 잘라냄
        if x < 0: 
            img_sticker = img_sticker[:, -x:]
            x = 0
        if y < 0:
            img_sticker = img_sticker[-y:, :]
            y = 0

        img_sticker = cv2.resize(img_sticker, (w,h)) # 스티커 이미지 조정
        # 스티커 부착할 영역을 선택하고 투명도를 조절후 부착
        sticker_area = img[y:y+img_sticker.shape[0], x:x+img_sticker.shape[1]]
        add_W_sticker = cv2.addWeighted(sticker_area, 1-img_Transparency, img_sticker, img_Transparency, 0)
        img[y:y+img_sticker.shape[0], x:x+img_sticker.shape[1]] = \
        np.where(img_sticker!=0,sticker_area,add_W_sticker).astype(np.uint8)
        
        return img
```
위와같이 주석으로 코드 구문마다 설명해주어서 코드를 이해하는데 어려움이 전혀 없었습니다.

### **[x] 코드가 에러를 유발할 가능성이 있나요?**
   - 없습니다.

### **[o] 코드 작성자가 코드를 제대로 이해하고 작성했나요?** (직접 인터뷰해보기)
   - 모두 이해하고 작성했습니다.
### **[x] 코드가 간결한가요?**
   - ``` detector_hog = dlib.get_frontal_face_detector() ``` 이 부분이 함수 안에 있어 함수 호출시 마다 모델을 메모리에 로드 하는 작업이 발생합니다.
   - 위 부분을 함수 밖으로 빼내어 함수 사용시에 이미 로드된 detector를 이용하면 불필요한 메모리 사용을 방지하고 실행속도를 올릴 수 있을 것입니다.

----------------------------------------------
### **참고 링크 및 코드 개선**
* 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
   - gpt와 직접 실험을 통해 ``` detector_hog = dlib.get_frontal_face_detector() ```의 실행 속도등을 확인 하였습니다.
* 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.

----------------------------------------------
