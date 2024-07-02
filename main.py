from PIL import Image
from tensorflow.keras.models import load_model  # 저장된 모델 불러오기
import numpy as np

filepath = input("예측할 숫자 파일명을 입력해주세요: ")

image = Image.open(filepath)  # 이미지 로드
image = image.convert('L')  # 흑백 사진으로 인식하게 함
image = image.resize((28, 28))  # 이미지를 모델이 인식할 수 있는 크기로 바꿔줌

data = np.array(image.getdata())  # 이미지를 numpy array로 변환
data = data.reshape(1, 28 * 28)  # 모델이 인식할 수 있게 쭉 펴준 뒤, 배치를 추가함.
data = data.astype('float32') / 255.0  # 0 ~ 1 사이의 값으로 스케일링

 # 모델은 흰색을 1로, 검은색으로 0으로 인식함.
 # 흰 바탕에 검은 붓으로 그림을 그렸다면 모델은 이를 반대로 인식을 하게 됨.
 # 따라서 모델이 인식할 수 있게 색을 반전시켜줌.
data = 1 - data 

model = load_model("model.h5")


pred_probs = model.predict(data)  # 클래스 확률 예측
pred = np.argmax(pred_probs, axis=-1)[0]  # 가장 높은 확률을 가지는 클래스 선택

# 실제 클래스와 예측된 클래스 출력
print("predict:", pred)  
