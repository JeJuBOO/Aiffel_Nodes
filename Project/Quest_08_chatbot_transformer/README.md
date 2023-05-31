# 아이펠캠퍼스 온라인4기 피어코드리뷰[23.05.31]

- 코더 : 부석경
- 리뷰어 : 김다인

---------------------------------------------
## **PRT(PeerReviewTemplate)**

### **[⭕] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?**
  - 모든 코드가 정상적으로 동작하고 질답이 잘 진행되었습니다   
  
   <img width="255" alt="image" src="https://github.com/JeJuBOO/Aiffel_Nodes/assets/94978101/deafbca6-a827-48f2-b6b3-0e78ae6af131">

### **[⭕] 주석을 보고 작성자의 코드가 이해되었나요?**
  -참고자료들을 링크해주고 중요한 부분들에 주석을 잘 달았습니다   
 <img width="383" alt="image" src="https://github.com/JeJuBOO/Aiffel_Nodes/assets/94978101/5432c066-5ca0-4d99-b060-07d9dde89551">
```python
def preprocess_sentence(sentence):
    # 문장의 앞뒤 공백을 제거
    sentence = sentence.strip()
    
    # 단어와 구두점(punctuation) 사이의 거리를 만듭니다.
    # 예를 들어서 "12시 땡!" => "12시 땡 !"와 같이
    # 땡와 느낌표(구두점) 사이에 거리를 만듭니다. 
    # (r"")에서 r은 raw line. 예시) '12시 땡!\n'까지 나타내는 표현식
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    # 공백이 연속으로 있을 때 하나로.
    sentence = re.sub(r'[" "]+', " ", sentence)
    # (ㄱ-ㅎ,가-힣,a-z, A-Z, ".", "?", "!", ",")를 제외한 모든 문자를 공백인 ' '로 대체
    sentence = re.sub(r'[^0-9ㄱ-ㅎ가-힣a-zA-Z.?!,]+', " ", sentence)
    # 숫자들을 모두 1로 통일 해보았습니다.
    sentence = re.sub(r'[0-9]+', "1", sentence)
    return sentence.strip()
```

### **[❌] 코드가 에러를 유발할 가능성이 있나요?**

### **[⭕] 코드 작성자가 코드를 제대로 이해하고 작성했나요?** (직접 인터뷰해보기)
  -코드를 잘 이해했습니다

### **[⭕] 코드가 간결한가요?**
```python
  - #질문과 답변의 쌍인 데이터셋을 구성하기 위한 데이터 로드 함수
def load_conversations(data):
    # Q의 중복되는 문장 제거.
    data.drop_duplicates(subset = ['Q'], inplace=True)
    
    question = list(map(preprocess_sentence,data['Q']))
    answer = list(map(preprocess_sentence,data['A']))

    return question, answer
    -map을 활용해 for문으로 복잡하게 구성하지 않고 간결하게 표현했습니다
```
----------------------------------------------
### **참고 링크 및 코드 개선**
* 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
* 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```python
# 질문과 답변의 쌍인 데이터셋을 구성하기 위한 데이터 로드 함수
def load_conversations(data):
    # Q의 중복되는 문장 제거.
    data.drop_duplicates(subset = ['Q'], inplace=True)
  ➡️질문은 같지만 대답이 다른 문장들이 있으므로 굳이 중복 질문을 제거하지 않아도 될 것 같습니다
```
----------------------------------------------
