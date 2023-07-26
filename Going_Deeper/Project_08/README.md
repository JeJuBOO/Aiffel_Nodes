# 아이펠캠퍼스 온라인4기 피어코드리뷰

- 코더 : 부석경
- 리뷰어 : 김설아

---------------------------------------------
## **PRT(PeerReviewTemplate)**

### **[⭕] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**

### **[⭕] 주석을 보고 작성자의 코드가 이해되었나요?**
```python
#   정사각형으로 잘라서 사용.
    def crop_roi(self, image, features, margin=0.2):
        img_shape = tf.shape(image)
        img_height = img_shape[0]
        img_width = img_shape[1]
        img_depth = img_shape[2]

        keypoint_x = tf.cast(tf.sparse.to_dense(features['image/object/parts/x']), dtype=tf.int32)
        keypoint_y = tf.cast(tf.sparse.to_dense(features['image/object/parts/y']), dtype=tf.int32)
        center_x = features['image/object/center/x']
        center_y = features['image/object/center/y']
        body_height = features['image/object/scale'] * 200.0
        
        # keypoint 중 유효한값(visible = 1) 만 사용합니다.
        masked_keypoint_x = tf.boolean_mask(keypoint_x, keypoint_x > 0)
        masked_keypoint_y = tf.boolean_mask(keypoint_y, keypoint_y > 0)
        
        # min, max 값을 찾습니다.
        keypoint_xmin = tf.reduce_min(masked_keypoint_x)
        keypoint_xmax = tf.reduce_max(masked_keypoint_x)
        keypoint_ymin = tf.reduce_min(masked_keypoint_y)
        keypoint_ymax = tf.reduce_max(masked_keypoint_y)
        
        # 높이 값을 이용해서 x, y 위치를 재조정 합니다. 박스를 정사각형으로 사용하기 위해 아래와 같이 사용합니다.
        xmin = keypoint_xmin - tf.cast(body_height * margin, dtype=tf.int32)
        xmax = keypoint_xmax + tf.cast(body_height * margin, dtype=tf.int32)
        ymin = keypoint_ymin - tf.cast(body_height * margin, dtype=tf.int32)
        ymax = keypoint_ymax + tf.cast(body_height * margin, dtype=tf.int32)
        
        # 이미지 크기를 벗어나는 점을 재조정 해줍니다.
        effective_xmin = xmin if xmin > 0 else 0
        effective_ymin = ymin if ymin > 0 else 0
        effective_xmax = xmax if xmax < img_width else img_width
        effective_ymax = ymax if ymax < img_height else img_height
        effective_height = effective_ymax - effective_ymin
        effective_width = effective_xmax - effective_xmin

        image = image[effective_ymin:effective_ymax, effective_xmin:effective_xmax, :]
        new_shape = tf.shape(image)
        new_height = new_shape[0]
        new_width = new_shape[1]
        
        effective_keypoint_x = (keypoint_x - effective_xmin) / new_width
        effective_keypoint_y = (keypoint_y - effective_ymin) / new_height
        
        return image, effective_keypoint_x, effective_keypoint_y
```
이해가 잘 되었습니다.
### **[❌] 코드가 에러를 유발할 가능성이 있나요?**
```python
class Preprocessor(object):
    def __init__(self,
                 image_shape=(256, 256, 3),
                 heatmap_shape=(64, 64, 16),
                 is_train=False):
        self.is_train = is_train
        self.image_shape = image_shape
        self.heatmap_shape = heatmap_shape

    def __call__(self, example):
        features = self.parse_tfexample(example)
        image = tf.io.decode_jpeg(features['image/encoded'])

        if self.is_train:
            random_margin = tf.random.uniform([1], 0.1, 0.3)[0]
            image, keypoint_x, keypoint_y = self.crop_roi(image, features, margin=random_margin)
            image = tf.image.resize(image, self.image_shape[0:2])
        else:
            image, keypoint_x, keypoint_y = self.crop_roi(image, features)
            image = tf.image.resize(image, self.image_shape[0:2])

        image = tf.cast(image, tf.float32) / 127.5 - 1
        heatmaps = self.make_heatmaps(features, keypoint_x, keypoint_y, self.heatmap_shape)

        return image, heatmaps

```
객체 형태로 조합하여 에러 유발 가능성을 없애셨습니다.
### **[⭕] 코드 작성자가 코드를 제대로 이해하고 작성했나요?** (직접 인터뷰해보기)
```python
    def compute_loss(self, labels, outputs):
        loss = 0
        weights = tf.cast(labels > 0, dtype=tf.float32) * 81 + 1
        if self.is_baseline:
            loss = tf.math.reduce_mean(
                tf.math.square(labels - outputs) * weights) * (
                    1.0 / self.global_batch_size)
        else:
            for output in outputs:
                loss += tf.math.reduce_mean(
                    tf.math.square(labels - output) * weights) * (
                        1.0 / self.global_batch_size)
        return loss
```
두 모델의 차이점에 대해 알고 코드를 수정하여 결과를 도출할 수 있도록 하셨습니다.
### **[⭕] 코드가 간결한가요?**
```python
HOURGLASS_WEIGHTS_PATH = os.path.join(PROJECT_PATH, 'model-epoch-2-loss-1.3278.h5')
hourglass_model = StackedHourglassNetwork(IMAGE_SHAPE, 4, 1)
hourglass_model.load_weights(HOURGLASS_WEIGHTS_PATH)

hourglass_image, hourglass_keypoints = predict(hourglass_model, test_image)
draw_keypoints_on_image(hourglass_image, hourglass_keypoints, index=None)
draw_skeleton_on_image(hourglass_image, hourglass_keypoints, index=None)
```
간결한 정리로 모델의 성능을 비교하셨습니다.
----------------------------------------------
### **참고 링크 및 코드 개선**
* 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
* 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.

----------------------------------------------
