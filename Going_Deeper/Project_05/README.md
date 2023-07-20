# 아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 부석경
- 리뷰어 : 심재형

PRT(PeerReviewTemplate)
----------------------------------------------

### 코드가 정상적으로 동작하고 주어진 문제를 해결했나요? (O)
![image](https://github.com/JeJuBOO/Aiffel_Nodes/assets/65104209/5e36539f-8c59-4a70-9ea9-3e4bee8cf9b2)

<br>결과또한 완벽하게 뽑았고 정리까지 완벽합니다. 주어진 문제를 해결하고 코드또한 정상적으로 작동해요!

![image](https://github.com/JeJuBOO/Aiffel_Nodes/assets/65104209/addc80c3-b486-49cf-bcec-db0a3aad34ed)

<br> Epoch 초반엔 IoU가 낮게 나타났으나 올릴수록 성능이 높은걸 보니 좋네요!
<br> 다만 Overfitting부분을 확인하신다면 모델의 학습도를 더 잘 파악할 수 있기 때문에 한다면 더 좋은 방법일 수 있을거같습니다!
<br> **history**부분을 추가함으로써 running curve를 그려보는것도 모델 학습도의 시각화를 빠르게 인지할 수 있을거같아요!

### 주석을 보고 작성자의 코드가 이해되었나요? (O)
```python
class KittiGenerator(tf.keras.utils.Sequence):
    '''
    KittiGenerator는 tf.keras.utils.Sequence를 상속받습니다.
    우리가 KittiDataset을 원하는 방식으로 preprocess하기 위해서 Sequnce를 커스텀해 사용합니다.
    '''
    def __init__(self, 
                   dir_path,
                   batch_size=16,
                   img_size=(224, 224, 3),
                   output_size=(224, 224),
                   is_train=True,
                   augmentation=None):
        '''
        dir_path: dataset의 directory path입니다.
        batch_size: batch_size 입니다.
        img_size: preprocess에 사용할 입력이미지의 크기입니다.
        output_size: ground_truth를 만들어주기 위한 크기입니다.
        is_train: 이 Generator가 학습용인지 테스트용인지 구분합니다.
        augmentation: 적용하길 원하는 augmentation 함수를 인자로 받습니다.
        '''
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.is_train = is_train
        self.dir_path = dir_path
        self.augmentation = augmentation
        self.img_size = img_size
        self.output_size = output_size

        # load_dataset()을 통해서 kitti dataset의 directory path에서 라벨과 이미지를 확인합니다.
        self.data = self.load_dataset()

    def load_dataset(self):
        # kitti dataset에서 필요한 정보(이미지 경로 및 라벨)를 directory에서 확인하고 로드하는 함수입니다.
        # 이때 is_train에 따라 test set을 분리해서 load하도록 해야합니다.
        # glob(dir) dir의 조건과 일치하는 파일명 전부 불러오기
        input_images = glob(os.path.join(self.dir_path, "image_2", "*.png"))
        label_images = glob(os.path.join(self.dir_path, "semantic", "*.png"))
        input_images.sort()
        label_images.sort()
        assert len(input_images) == len(label_images) # 입력과 라벨의 개수가 일치하는가 확인
        data = [ _ for _ in zip(input_images, label_images)]

        if self.is_train:
            return data[:-30] # 훈련집합이면 뒤 30개을 제외한 170개를 return
        
        return data[-30:] # 테스트 집합은 30개

```
각 항목별로 주석을 잘 달아주셨고 이해가 굉장히 빨리됩니다!

### 코드가 에러를 유발할 가능성이 있나요? (X)
```python
# Skip conn
    ux1_0 = UpSampling2D(2)(x1_0)
    cx0_1 = concatenate([x0_0, ux1_0], axis=3)
    x0_1 = encoder_block(cx0_1, 64)
    
    ux2_0 = UpSampling2D(2)(x2_0)
    cx1_1 = concatenate([x1_0, ux2_0], axis=3)
    x1_1 = encoder_block(cx1_1, 128)
    
    ux3_0 = UpSampling2D(2)(x3_0)
    cx2_1 = concatenate([x2_0, ux3_0], axis=3)
    x2_1 = encoder_block(cx2_1, 256)
    
    ux1_1 = UpSampling2D(2)(x1_1)
    cx0_2 = concatenate([x0_0, x0_1, ux1_1], axis=3)
    x0_2 = encoder_block(cx0_2, 64)
    
    ux2_1 = UpSampling2D(2)(x2_1)
    cx1_2 = concatenate([x1_0, x1_1, ux2_1], axis=3)
    x1_2 = encoder_block(cx1_2, 128)
    
    ux1_2 = UpSampling2D(2)(x1_2)
    cx0_3 = concatenate([x0_0, x0_1, x0_2, ux1_2], axis=3)
    x0_3 = encoder_block(cx0_3, 64)
```
<br>완벽한 skip conn을 구현했고 에러가 날 부분은 보이지 않습니다 GOOD!

### 코드 작성자가 코드를 제대로 이해하고 작성했나요? (O)
![image](https://github.com/JeJuBOO/Aiffel_Nodes/assets/65104209/f9f1039a-f415-43eb-8276-6269115bed32)

<br>하나의 코드로 U-Net과 U-Net++를 동시에 나타나개 함으로써 동작방식을 정확히 이해하고있습니다!
### 코드가 간결한가요? (O)
![image](https://github.com/JeJuBOO/Aiffel_Nodes/assets/65104209/0313921c-47c8-452d-b823-5df5bc561051)


<br> 대부분의 함수처리로 깔끔한 코드로 만들어서 직관적으로 보기 편합니다!

----------------------------------------------
