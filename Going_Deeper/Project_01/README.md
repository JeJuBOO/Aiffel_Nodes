# 아이펠캠퍼스 온라인4기 피어코드리뷰[23.06.26]

- 코더 : 부석경
- 리뷰어 : 이성주

---------------------------------------------
## **PRT(PeerReviewTemplate)**

### **[⭕] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
|평가문항|상세기준|완료여부|
|-------|---------|--------|
|ResNet-34, ResNet-50 모델 구현이 정상적으로 진행되었는가?|블록함수 구현이 제대로 진행되었으며 구현한 모델의 summary가 예상된 형태로 출력되었다.| ![image](https://github.com/JeJuBOO/Aiffel_Nodes/assets/29011595/d7413e23-e752-427e-8924-e604e5d8bd80)  ResNet 블록 구현 코드를 잘 작성하였습니다.|
|구현한 ResNet 모델을 활용하여 Image Classification 모델 훈련이 가능한가?|tensorflow-datasets에서 제공하는 cats_vs_dogs 데이터셋으로 학습 진행 시 loss가 감소하는 것이 확인되었다.|![image](https://github.com/JeJuBOO/Aiffel_Nodes/assets/29011595/7c2b1010-5a17-4a81-898a-f616c370095a)  제작한 모델로 Image Classification 모델 훈련시 loss가 감소 되는 것을 확인하여 Image Classification 모델 훈련이 가능합니다. |
|Ablation Study 결과가 바른 포맷으로 제출되었는가?|ResNet-34, ResNet-50 각각 plain모델과 residual모델을 동일한 epoch만큼 학습시켰을 때의 validation accuracy 기준으로 Ablation Study 결과표가 작성되었다 |![image](https://github.com/JeJuBOO/Aiffel_Nodes/assets/29011595/42601abc-e94e-4d27-842a-0aea788397f3) 각 모델들을 동일 에폭으로 학습하여 val acc를 비교한 결과표가 작성되었습니다.|
### **[⭕] 주석을 보고 작성자의 코드가 이해되었나요?**
![image](https://github.com/JeJuBOO/Aiffel_Nodes/assets/29011595/9d0e96c9-0ab6-44ff-928f-de090a3c4c79)
 - 주석을 보고 코드가 무엇을 의미하는지 이해가 되었습니다.

### **[❌] 코드가 에러를 유발할 가능성이 있나요?**
 - 위 주석 사진과 같이 OOM를 방지하기 위해 datasets 조절도 하며, 코드가 에러를 유발할 가능성이 없어 보입니다.
### **[⭕] 코드 작성자가 코드를 제대로 이해하고 작성했나요?** (직접 인터뷰해보기)
 - Resnet 구조를 잘 이해하고 작성했습니다.!
### **[⭕] 코드가 간결한가요?**
![image](https://github.com/JeJuBOO/Aiffel_Nodes/assets/29011595/d730bffb-0ae7-4be9-853f-413f705b1c79)
 - is_50과 is_plain을 한 함수에 적용하여 코드를 간결하게 작성하였습니다.
----------------------------------------------
### **참고 링크 및 코드 개선**
* 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
* 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.

----------------------------------------------
