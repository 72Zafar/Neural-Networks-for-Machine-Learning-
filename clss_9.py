from  sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np 

hours_student = [2.5,1.5,3.0, 1.8,4.0,2.0,3.5,2.7]
prev_exam_score = [80,70,7.5,60,85,80,90,65,]
exam_score = ["pass","Fail","pass","Fail","pass","pass","pass","Fail"]

label_encoder =LabelEncoder()
encoder_exam_score = label_encoder.fit_transform(exam_score)

x = np.column_stack((hours_student,prev_exam_score))
y = encoder_exam_score

clf = MLPClassifier(hidden_layer_sizes=(4,),activation="logistic",max_iter=1000,random_state=42)
clf.fit(x,y)

new_student_data = np.array([[1.1,11]])

predicted_outcome  = clf.predict(new_student_data)

predicted_outcome_decode = label_encoder.inverse_transform(predicted_outcome)

print ("predicted Exam outcome for the new student {}".format(predicted_outcome_decode[0]))





