# 과제개요
전천후 작전수행을 위한 화상 이미지 노이즈 제거
(Restoration, Denoising)    
이미지 영역|PSNR

문제 정의
주어진 이미지에 대하여 안개, 우천 등 기상상황으로 인하여 저하된 품질을 복원시키는 문제

---

# 평가지표
PSNR (Peak Singal-to-Noise Ratio)

---

# 데이터셋
위성으로 촬영한 안개, 비 등의 노이즈가 추가된 이미지 데이터셋 
RGB 분포중 초록색, 연갈색의 비중이 매우 높음
Image Size (512x512)

---

# 학습 내용
### 모델 선정
- 베이스라인 (Transformer 계열([Restormer](https://github.com/swz30/Restormer)))
  - 데이터 셋 양이 적어 트렌스포머 계열 모델학습에 맞지않음
  - 모델 학습 특성상 시간이 너무 오래걸림
    - 대회기간이 하루밖에 되지않기 때문에 시간적 제약이 존재

- 베이스라인을 사용하지 않고 Resnet계열 모델([PRENet](https://github.com/csdwren/PReNet))로 변경
  - paper with code 를 보면 Sota모델(Transformer계열)과 이전 모델의 성능차이가 크게 커 보이지 않았음
  - 학습 속도가 빠르고 간편한 모델을 선정
    - 딥러닝이 아닌 머신러닝으로도 준수한 성능이 나올 정도의 Task
    - 다양한 실험을 적용해보기 위해 
  

### AMP 적용
- 학습 속도 향상을 위해 AMP 적용

### 학습
- resize
  - 학습 시 이미지를 256x256으로 리사이즈
  - 추론시에는 512x512사이즈로 추론 (Interpolation이 큰 영향을 준다고 생각함)
- Loss 선택 -> 22.2 PSNR
  - 평가지표에 맞는 Loss를 사용 => SIMM Loss v.s. PSNR Loss
  - 결과를 비교해 보았을 때 PSNR Loss가 더 좋은 성능을 보였음

### Cut blur
- SR, Denoised task에 효과적인 augmentation
- 초기에는 좋은 성능을 보였으나 결과적으로 부정적인 영향을 끼침

### Post-Processing -> 24.3 PSNR 
- color shift
  - 예측된 결과에서 짙은 초록색이 많이 검출되어 이를 보완하기 위해 사용
  - 색 범위를 활용하여 마스킹을 진행하고 해당 영역의 밝기를 키움
- hsv shift
  - 학습 과정에서 hsv 채널 분석 중 값이 편향됨을 확인하여 이를 fix하기위해 hsv shift 진행
- Normaliziation
  - Output 이미지의 히스토그램을 분석하여 이미지 정규화 진행
  - 이미지의 노이즈가 특정 영역에 몰려 있는 경우 화질을 개선하기 위해 주로 사용됨
- 빗방울 제거
  - Output 데이터를 분석한 결과 빗방울과 나머지 배경을 비교하였을 때 빗방울이 높은 밝기를 갖고 있어 특정값 이상의 값만 밝기를 낮춤
---
# 코드 내용

### 핵심 파일 설명

- EDA 코드 : `eda_process/data_eda.ipynb`
- 데이터셋 구축 및 전처리 코드 : `code/CustomDataset.py`
- 학습 코드 : `code/train_PReNet.py`
  - 사용법 : `python code/train_PReNet.py --preprocess False --save_path $MODEL_PATH --data_path $DATA_PATH`
- 추론 코드 : `code/test_PReNet.py`
  - 사용법 : `python code/test_PReNet.py --logdir $MODEL_PATH --data_path $TEST_DIR --save_path $RESULT_SAVE_DIR`
- 시각화 코드 : `eda_process/image_diff.ipynb`
- 후처리 코드 : `eda_process/Postprocess.ipynb`
- 앙상블 코드 : `eda_process/Ensemble.ipynb`

### 학습 과정 설명
1. EDA 코드를 활용하여 데이터 분석
2. PReNet 학습
  - `python code/train_PReNet.py --preprocess False --save_path $MODEL_PATH --data_path $DATA_PATH`
3. 학습 중 저장된 모델(pth)로 test데이터 추론 및 시각화
  - `python code/test_PReNet.py --logdir $MODEL_PATH --data_path`
  - `eda_process/image_diff.ipynb` 활용
4. 추론 결과 분석 및 후처리
  - `da_process/Postprocess.ipynb` 활용
5. 결과물 앙상블 -> 25.1 PSNR
  - `eda_process/Ensemble.ipynb` 활용

---

# 학습 결과 및 검증
- Ansemble한 Pretrained model은 `/workspace/Final_Submission/PReNet/logs/real/` 내에 모두 존재
  - net_iter1601.pth : SSIMLoss, batch_size=20, iteration=1600를 사용한 모델
    - 결과 : results/ee1
  - net_iter2001.pth : SSIMLoss, batch_size=20, iteration=2000를 사용한 모델
    - 결과 : results/prenet2000_shift
    - 모델을 통해 나온 결과에 color shift 적용 (test_real.py 참고)
  - net_iter3001.pth : SSIMLoss, batch_size=20, iteration=3000를 사용한 모델
    - 결과 : results/3000pth_norm
    - 모델을 통해 나온 결과에 Normalization 적용 (Postprocess.ipynb 참고)
<!-- - Model 추론 방법
  - test_PReNet.py --logdir $MODEL_PATH --data_path $TEST_DIR --save_path $RESULT_SAVE_DIR -->
- Model Ensemble
  - 위 결과 3가지 모델에 대하여 앙상블 진행
  - soft ensemble
  - `Ensemble.ipynb` 참고

---

### using module
- opencv-python=4.6.0.66
- pytorch=1.12.0
- torchvision=0.13.0 
- scikit-image=0.19.2
- scikit-learn=1.0.2         
- python=3.9.12        
- numpy=1.21.5    

# 실행 환경 설정

  - 실행 환경 
    ```
    conda create -n wicd python=3.9.12
    conda activate wicd
    conda install numpy=1.21.5
    conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
    pip install opencv-python==4.6.0.66
    pip install scikit-image==0.19.2
    pip install scikit-learn==1.0.2
    pip install h5py
    pip install albumentations
    pip install GPUtil
    ```
    ### using module
  - opencv-python=4.6.0.66
  - pytorch=1.12.0
  - torchvision=0.13.0 
  - scikit-image=0.19.2
  - scikit-learn=1.0.2         
  - python=3.9.12        
  - numpy=1.21.5   
 
# 학습 및 테스트 실행 방법

- 데이터셋 폴더 생성 및 파일 넣기
  - datasets 폴더 생성
  - datasets 폴더 내에 Noised 및 Denoised 폴더 생성
  - Noised와 Denoised 폴더에 input 데이터 및 target 데이터를 넣기

- 학습 한 모델 저장소 생성
  - logs 폴더 생성
  - logs 폴더 내에 PReNet 폴더 생성

- 학습 코드 실행
  ```
   sh train_PReNet.sh
  ```

  - 테스트 코드 실행
  ```
  sh test_PReNet.sh
  ```
---

# 추가적인 시도
- 데이터를 두개로 분류한 후 각각의 모델을 돌리고 앙상블을 시도
  - 데이터를 뜯어본 결과 크게 초록색이 많이 포함된 배경과 그렇지 않은 배경으로 나눌 수 있었음
  - 각각의 모델을 돌리고 앙상블을 했으나 성능이 최종제출본보다 낮았음

- 특정 알고리즘 패턴 파악 후 채널 지정
  - 해당 빗물이나 안개가 일정한 알고리즘 패턴에 의해서 작성된것이 보였음.
  - 그 패턴을 만들어낸 필터 사이즈를 찾아 해당 필터 사이즈로 학습을 처리하면 성능이 높아질거라 예상.
  - 3x3 채널로 노이즈가 입혀졌다는건 파악했지만 시간 부족으로 더 이상 진행하지 못했음.
 
---

# 팀 회고

### 잘한점
- EDA를 통해 다음 단계를 분석하여 진행
  - 모델의 특성을 파악하며 후처리 코드 작성
  - 학습 결과 및 후처리 결과를 눈으로 확인, 자체적인 평가 지표를 통해 성능향상을 도모
- 팀의 분위기
  - 의사결정이 정확하고 근거가 있었으며 신속했다.
  - 무박 2일임에도 불구하고 텐션유지
  - 역할 분배가 깔끔하게 이루어졌으며 협업 또한 원활

### 부족했던점 (아쉬운점)
- 추가적인 어텐션 맵을 사용하지 않고 빗방울에 대한 특징을 뽑아내기에는 한계가 있었다. 이에 대한 테스트를 못해봐서 아쉬움
- 초기 서버 설정에서 오류가 발생하여 시간을 너무 많이 소모했다.
- 한 가지 모델만 사용했다.
  - 준비한 모델이 많았지만 이를 더 완벽히 준비해왔다면 다양한 모델을 이용해 볼 수 있었을 듯
- 대회 전 준비한 기법을 모두 사용해보지 못함
  - 앙상블 방법의 다양성 부족
  - 전처리
  - Augmentation
