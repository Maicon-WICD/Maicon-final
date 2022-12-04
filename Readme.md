# 학습 내용
### 모델 선정
- Transformer 계열(baseline) -> Resnet계열 모델 변경
  - 데이터 셋 양이 적어 트렌스포머 계열에 맞지않음
  - 다양한 실험을 적용해보기 위해 학습 속도가 빠르고 간편한 모델을 선정

### AMP 적용
- 학습 속도 향상을 위해 AMP 적용

### 학습 과정
- resize
  - 학습 시 이미지를 256x256으로 리사이즈
  - 추론시에는 512x512사이즈로 추론
- Loss 선택
  - 평가지표에 맞는 Loss를 사용 => SIMM Loss v.s. PNSR Loss
  - 결과를 비교해 보았을 때 SIMM Loss가 더 좋은 성능을 보였음

### Cut blur
- SR, Denoised task에 효과적인 augmentation
- 초기에는 좋은 성능을 보였으나 결과적으로 부정적인 영향을 끼침

### Post-Processing
- color shift
  - 예측된 결과에서 짙흔 초록색이 많이 검출되어 이를 보완하기 위해 사용
  - 색 범위를 활용하여 마스킹을 진행하고 해당 영역의 밝기를 키움
- hsv shift
  - 학습 과정에서 hsv 채널 분석중 값이 편향됨을 확인하여 이를 fix하기위해 hsv shift 진행
- Normaliziation
  - Output 이미지의 히스토그램을 분석하여 이미지 정규화 진행
  - 이미지의 노이즈가 특정 영역에 몰려 있는 경우 화질을 개선하기 위해 주로 사용됌

# 학습 결과 및 검증
- Ansemble한 Pretrained model은 /workspace/Final_Submission/PReNet/logs/real/ 내에 모두 존재
  - net_iter1601.pth : SSIMLoss, batch_size=20, iteration=1600를 사용한 모델
    - 결과 : results/ee1
  - net_iter2001.pth : SSIMLoss, batch_size=20, iteration=2000를 사용한 모델
    - 결과 : results/prenet2000_shift
    - 모델을 통해 나온 결과에 color shift 적용 (test_real.py 참고)
  - net_iter3001.pth : SSIMLoss, batch_size=20, iteration=3000를 사용한 모델
    - 결과 : results/3000pth_norm
    - 모델을 통해 나온 결과에 Normalization 적용 (Postprocess.ipynb 참고)
- Model 추론 방법
  - test_PReNet.py --logdir $MODEL_PATH --data_path $TEST_DIR --save_path $RESULT_SAVE_DIR
- Model Ensemble
  - 위 결과 3가지 모델에 대하여 앙상블 진행
  - soft ensemble
  - Ensemble.ipynb 참고

### using module
- opencv-python=4.6.0.66
- pytorch=1.12.0
- torchvision=0.13.0 
- scikit-image=0.19.2         s
- scikit-learn=1.0.2         
- python=3.9.12        
- numpy=1.21.5    
