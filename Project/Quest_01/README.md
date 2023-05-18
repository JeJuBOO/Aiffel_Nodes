# 아이펠캠퍼스 온라인4기 피어코드리뷰 [23.05.18]

- 코더 : 부석경
- 리뷰어 : 이성주

---------------------------------------------
## **PRT(PeerReviewTemplate)**

### [o] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?   

``` python
test_pred = model(X_test, W, b)
test_mse = MSE(test_pred,y_test)

print("test데이터의 MSE값은 : ",test_mse)
```
test데이터의 MSE값은 :  2839.6180205127453

```python
from sklearn.metrics import mean_squared_error

test_pred = model.predict(X_test)
MSE = mean_squared_error(test_pred, y_test)
RMSE = MSE**0.5

print(f" MSE 오차는 : {MSE}\n RMSE 오차는 : {RMSE}")
```
 MSE 오차는 : 20619.64878501189   
 RMSE 오차는 : 143.5954344156244

### [o] 주석을 보고 작성자의 코드가 이해되었나요?
``` python
from sklearn.model_selection import train_test_split
# test_size는 훈련집합과 테스트집합의 비율로(0.8:0.2)을 의미
# random_state는 랜덤으로 훈련집합과 테스트집합을 나누기 때문에 반복 수행마다 
#   달라지는 것을 방지하기  위한 랜덤의 seed번호이다. 
X_train, X_test, y_train, y_test = \
    train_test_split(df_X, df_y,test_size=0.2,random_state=41)

print('훈렵집합과 테스트집합의 입력값의 shape\n',X_train.shape,X_test.shape)
print('훈렵집합과 테스트집합의 정답(label)값의 shape\n',y_train.shape,y_test.shape)
```
``` python
model = LinearRegression() # 모델 불러오기
model.fit(X_train, y_train) # 모델 학습하기
```
주석으로 알맞게 코드를 설명해줘서 이해하기에 편하였습니다.
### [v] 코드가 에러를 유발할 가능성이 있나요?
+ 없습니다.

### [o] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
+ 해당 프로젝트의 진행사항을 모두 이해하고 있었습니다.
### [o] 코드가 간결한가요?
+ Projet01 : clean한 코드 작성이였습니다.
+ Projet02 
``` python
train_df = train_df.drop('casual',axis = 'columns') # 미등록 사용자 대여수
train_df = train_df.drop('registered',axis = 'columns') # 등록 사용자 대여수
train_df = train_df.drop('datetime',axis = 'columns') 
train_df = train_df.drop('second',axis = 'columns') # 모두 0
train_df = train_df.drop('minute',axis = 'columns') # 모두 0
train_df = train_df.drop('count',axis = 'columns') # 정답 데이터
```
 위 코드를 한줄로 표현 할 수 있습니다. 
``` python
train_df = train_df.drop(['datetime', 'casual', 'registered', 'count'], axis=1)
```

----------------------------------------------
### 참고 링크 및 코드 개선
* 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
* 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
``` python
pd.read_csv('g:/내 드라이브/Aiffel_Nodes/data_preprocess/bike_sharing_demand/train.csv')
# 파일경로를 한글보다 영어를 사용하는 것이 더 바람직해 보입니다.
```
----------------------------------------------
