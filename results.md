## Checked with my cropped image
### Result: on real time cropped images

#### On "me_open_eye.jpeg"
(Road-Accident-Risk-Detection) ➜  Road-Accident-Risk-Detection git:(main) ✗ python test_cnn.py
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
2025-06-11 21:34:35.169 Python[21076:4663634] +[IMKClient subclass]: chose IMKClient_Modern
2025-06-11 21:34:35.169 Python[21076:4663634] +[IMKInputSession subclass]: chose IMKInputSession_Modern
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 113ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
Pred: [[1.]]

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
predict: 1.0

Prediction: Eyes are Open

#### On "me_closed_eye.jpeg"
(Road-Accident-Risk-Detection) ➜  Road-Accident-Risk-Detection git:(main) ✗ python test_cnn.py 
   
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
2025-06-11 21:35:01.943 Python[21134:4664176] +[IMKClient subclass]: chose IMKClient_Modern
2025-06-11 21:35:01.943 Python[21134:4664176] +[IMKInputSession subclass]: chose IMKInputSession_Modern
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 72ms/step
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 30ms/step
Pred: [[3.1763592e-10]]

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 29ms/step
predict: 3.176359175682819e-10

Prediction: Eyes are Closed`