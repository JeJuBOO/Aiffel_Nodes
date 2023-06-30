# 아이펠캠퍼스 온라인4기 피어코드리뷰[23.]

- 코더 : 부석경
- 리뷰어 : 이동익

---------------------------------------------
## **PRT(PeerReviewTemplate)**

### **[⭕] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
|평가문항|상세기준|완료여부|
|-------|---------|--------|
|1. CAM을 얻기 위한 기본모델의 구성과 학습이 정상 진행되었는가?|ResNet50 + GAP + DenseLayer 결합된 CAM 모델의 학습과정이 안정적으로 수렴하였다.| O |
|2. 분류근거를 설명 가능한 Class activation map을 얻을 수 있는가?|CAM 방식과 Grad-CAM 방식의 class activation map이 정상적으로 얻어지며, 시각화하였을 때 해당 object의 주요 특징 위치를 잘 반영한다.| O |
|3. 인식결과의 시각화 및 성능 분석을 적절히 수행하였는가?|CAM과 Grad-CAM 각각에 대해 원본이미지합성, 바운딩박스, IoU 계산 과정을 통해 CAM과 Grad-CAM의 object localization 성능이 비교분석되었다.| O |

1. 시각화를 통해 수렴 결과를 확인할 수 있었습니다.  
![image](https://github.com/JeJuBOO/Aiffel_Nodes/assets/126870709/522e7e08-c4bf-4215-96d1-2f28e359e51b)
 
2. 레이어별로 CAM이 출력되었고, 강아지에 해당하는 부분이 활성화 된 것으로 보여집니다.    
![image](https://github.com/JeJuBOO/Aiffel_Nodes/assets/126870709/c027c9f7-1149-4832-9de2-7e1f18795a2d)

3. CAM과 Grad-CAM의 IoU 계산을 통해 성능을 비교할 수 있었고, 탐지 부분을 시각화하여 보여주셨습니다.   
![image](https://github.com/JeJuBOO/Aiffel_Nodes/assets/126870709/f3485a78-bdf0-4a21-9d7a-366cfee98f81)

### **[⭕] 주석을 보고 작성자의 코드가 이해되었나요?**
네 CAM 구현의 과정별로 이해가 잘 되었습니다.
```python
def generate_grad_cam(model, activation_layer, item):
    grad_cam_image = None
    item = copy.deepcopy(item)
    width = item['image'].shape[1]
    height = item['image'].shape[0]
    img_tensor, class_idx = normalize_and_resize_img(item)
    
    # 원하는 레이어를 선택할 수 있는 `activation_layer` 추가.
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(activation_layer).output, model.output])
    
    # `GradientTape`를 이용하여 모델의 그레이언트 얻기
    with tf.GradientTape() as tape:
        conv_output, pred = grad_model(tf.expand_dims(img_tensor, 0))
    
        loss = pred[:, class_idx] 
        output = conv_output[0] 
        grad_val = tape.gradient(loss, conv_output)[0]
    
    # 위에서 구한 그레이언트를 GAP으로 가중치 구하기
    weights = np.mean(grad_val, axis=(0, 1))
    
    # 가중치를 conv_output에 곱하고 모두 더한다.
    grad_cam_image = np.zeros(dtype=np.float32, shape=conv_output.shape[0:2])
    for k, w in enumerate(weights):
        grad_cam_image += w * output[:, :, k]
        
    # Relu, 0이하 값을 0으로
    grad_cam_image = tf.math.maximum(0, grad_cam_image)
    # 이미지 값에서 가장 큰값으로 나눠줌으로 정규화.
    grad_cam_image /= np.max(grad_cam_image)
    grad_cam_image = grad_cam_image.numpy()
    # 원래 이미지의 크기로 리사이즈
    grad_cam_image = cv2.resize(grad_cam_image, (width, height))
    
    return grad_cam_image

```

### **[❌] 코드가 에러를 유발할 가능성이 있나요?**

### **[⭕] 코드 작성자가 코드를 제대로 이해하고 작성했나요?** (직접 인터뷰해보기)
Grad-CAM에서 제안하는 방법, object detection을 위해 cv2함수를 이용하여 바운딩박스를 얻는 과정 등을 잘 이해하고 구현하신 것 같습니다.
### **[⭕] 코드가 간결한가요?**

원하는 레이어의 CAM을 모두 모아 출력하는 코드가 간편하고 좋아보입니다.
```python
def cam_plots(cam_model,activation_names,item):

    plt.figure(figsize=(20,5))

    ax = plt.subplot(1,len(activation_names)+1, 1)
    ax.imshow(item['image'])
    plt.title('input image')
    plt.axis("off")
    i = 2
    for act_name in activation_names:
        grad_cam_image = generate_grad_cam(cam_model, act_name, item)
        ax = plt.subplot(1,len(activation_names)+1, i)
        ax.imshow(grad_cam_image)
        plt.title(act_name)
        plt.axis("off")
        i += 1
```
----------------------------------------------
### **참고 링크 및 코드 개선**
* 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
* 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.

----------------------------------------------
