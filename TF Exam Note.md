# TF Exam Note
https://hackmd.io/08SSi1YeSp-k6TDV00T1vw?edit
###### tags: `tf_exam`
1. 在第四堂課第三週時，我用lstm建構模型時，要將最後結果畫出來時發生錯誤，我發現我必須將要繪制的資料做過特殊處理後(model_predict)才能畫出，這點要注意
2. Some advice:
* https://www.reddit.com/r/datascience/comments/gc29zj/passed_tensorflow_developer_certification/

* https://www.kaggle.com/questions-and-answers/196276
* https://medium.com/@rbarbero/tensorflow-certification-tips-d1e0385668c8
* sequence to sequence tensorflow window:
https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/discussions/weeks/4/threads/rimRoutYEem__xK0XuhC7A

## 檔案內容連結
### Tf_course1
* 簡單的線性代數預測Course1_part2_lesson2：
https://github.com/hao134/Tf_course/blob/main/Tf_course1/Course1_part2_lesson2.py
* 簡單的線性代數預測2 Course1_part4_lesson2：
https://github.com/hao134/Tf_course/blob/main/Tf_course1/Course1_Part4_Lesson2.py
* 用fashion mnist資料集，以自製的callback（在達到要求的accuracy時停止訓練 (callback):
https://github.com/hao134/Tf_course/blob/main/Tf_course1/Course1_part4_lesson4.py
* fashoion mnist, 以普通的全連結神經網路和Conv神經網路，比較結果：
https://github.com/hao134/Tf_course/blob/main/Tf_course1/Course1_Part6_Lesson2.py
* 真實數據集馬和人照片，用Conv神經網路，(image generator , conv):
https://hackmd.io/08SSi1YeSp-k6TDV00T1vw?edit
* Week2's exercise (mnist, callback):
https://github.com/hao134/Tf_course/blob/main/Tf_course1/Week2_exercise.py
* Week3's exercise (mnist, callback, conv):
https://github.com/hao134/Tf_course/blob/main/Tf_course1/Week3_exercise.py
* Week4's exercise (真實照片數據(笑臉/哭臉），callback, image generator, conv):
https://github.com/hao134/Tf_course/blob/main/Tf_course1/Week4_exercise.py

### Tf_course2
* (真實數據貓狗辨識，image generator, deal with data(train,validation不用自己分, train/validation):
https://github.com/hao134/Tf_course/blob/main/Tf_course2/Course2_part2_lesson2.py
* (真實數據馬人辨識，image generator,**data augmentation**,  deal with data(train,validation不用自己分, train/validation):
https://github.com/hao134/Tf_course/blob/main/Tf_course2/Course2_Part4_lesson4.py
* (真實數據貓狗辨識, **Transfer Learning**, 剩下同上)：
https://github.com/hao134/Tf_course/blob/main/Tf_course2/Course2_part6_lesson3.py
* (真實數據剪刀石頭布分類， **多類別分類任務**，剩下同上)：
https://github.com/hao134/Tf_course/blob/main/Tf_course2/Course2_part8_lesson2.py
* Week1's Exercise (真實數據貓狗辨識， **deal with data (要自己分train/validation)**)：
https://github.com/hao134/Tf_course/blob/main/Tf_course2/Week1's%20exercise.py
* Week2's Exercise (真實數據貓狗辨識，**deal with data (要自己分train/validation)**，**data with augmentation**):
https://github.com/lmoroney/dlaicourse/blob/master/Exercises/Exercise%206%20-%20Cats%20v%20Dogs%20with%20Augmentation/Exercise%206%20-%20Answer.ipynb
* Week3's Exercise (真實數據人馬辨識，**Transfer Learning**):
https://github.com/hao134/Tf_course/blob/main/Tf_course2/week3_exercise.py
* Week4's Exercise (真實數據號誌分類，**multi-output classification (26 classes)**)
https://github.com/hao134/Tf_course/blob/main/Tf_course2/Week4_exercise.py
Use Transfer learning on Week4's exercise (vgg16 perform well):
https://github.com/hao134/Tf_course/blob/main/Tf_course2/week4_exercise.ipynb

### Tf_course3
* 1.Week1_lesson1(Use *Tokenizer* on simple sentences)：
https://github.com/hao134/Tf_course/blob/main/Tf_course3/Week1_lesson1.py
* 2.Week1_lesson2(Use *Tokenizer* on simple sentences, *pad_sequences*):
https://github.com/hao134/Tf_course/blob/main/Tf_course3/Week1_lesson2.py
* 3.Week2_lesson1(資料*tfds* imdb_reviews, *Tokenizer*, *pad_sequences*, **Embedding**, Using simple dnn train it):
https://github.com/hao134/Tf_course/blob/main/Tf_course3/Week2_lesson1.py
* 4.Week2_lesson2(資料sarcasm(判斷語句是否諷刺)，deal with data(use json load data, split train/validation),其餘同3）:
https://github.com/hao134/Tf_course/blob/main/Tf_course3/Week2_lesson2.py
* 5.Week2_lesson3(資料*tfds* imdb_reviews/subwords8k，主要是看看subword是否有助於訓練)：
https://github.com/hao134/Tf_course/blob/main/Tf_course3/Week2_lesson3.py
* 6.Week3_lesson1a(資料*tfds* imdb_reviews/subwords8k, **different Model**):
(1). One layer Bidirectional LSTM:
https://github.com/hao134/Tf_course/blob/main/Tf_course3/Week3_lesson1a.py
(2). Two layers Bidirectional LSTM (add return_sequences = True):
https://github.com/hao134/Tf_course/blob/main/Tf_course3/Week3_lesson1b.py
(3). Convolution 1D layer:
https://github.com/hao134/Tf_course/blob/main/Tf_course3/Week3_lesson1c.py
* 7.Week3_lesson2(資料sarcasm, LSTM model, 其餘同3):
https://github.com/hao134/Tf_course/blob/main/Tf_course3/Week3_lesson2.py
* 8.Week3_lesson2c(資料sarcasm, conv1D model，其餘同3
https://github.com/hao134/Tf_course/blob/main/Tf_course3/Week3_lesson2c.py
* 9.Week3_lesson2d(資料imdb_reviews, GRU model, **deal with data (train/validation/test)**，其餘同3)
https://github.com/hao134/Tf_course/blob/main/Tf_course3/Week3_lesson2d.py
* 10.Week4_lesson1(資料詩詞（較短），根據詩詞分類字)：
https://github.com/hao134/Tf_course/blob/main/Tf_course3/Week4_lesson1.py
* 11.Week4_lesson2(資料詩詞（較長），根據詩詞分類字)：
https://github.com/hao134/Tf_course/blob/main/Tf_course3/Week4_lesson2.py

* Week1's Exercise(Use *Tokenizer* on bbc text, Banned the stopwords, Deal with data):
https://github.com/hao134/Tf_course/blob/main/Tf_course3/Week1's%20exercise.py
* Week2's Exercise(Train an Simple model on bbc text(6 classes classification task), 其餘同week1)
https://github.com/hao134/Tf_course/blob/main/Tf_course3/Week2_exercise.py
* Week3's Exercise(真實資料判斷是否，**用glove6b 來embedding**，deal with data(參考，很重要) )
https://nbviewer.jupyter.org/github/hao134/Tf_course/blob/main/Tf_course3/NLP_Course_Week_3_Exercise_Answer.ipynb
* Week4's Exercise(真實資料莎士比雅，創出莎士比雅風格文章，LSTM model with dropout)：
https://github.com/hao134/Tf_course/blob/main/Tf_course3/Week4_exercise.py

### Tf_course4
* 1.Week2_lesson1 (*tf.data.Dataset*, 各種window_dataset的產出方式）：
https://github.com/hao134/Tf_course/blob/main/Tf_course4/week2_lesson1.py
* 2.Week2_lesson2 (產出有時序性和噪音的資料，如何使用windowed_dataset，用很簡單的dnn預測)：
https://github.com/hao134/Tf_course/blob/main/Tf_course4/week2_lesson2.py
* 3.Week2_lesson3 (用稍微複雜的dnn model預測，learningratescheduler, 再用挑好的learning rate 預測)
https://github.com/hao134/Tf_course/blob/main/Tf_course4/Week2_lesson3.py
* 4.Week3_lesson2 (Use Simple RNN來預測先前的時序性資料，其餘同3):
https://github.com/hao134/Tf_course/blob/main/Tf_course4/Week3_lesson2.py
* 5.Week3_lesson4 (Use LSTM 來預測先前的時序性資料，其餘同3):
https://github.com/hao134/Tf_course/blob/main/Tf_course4/Week3_lesson4.py
* 6.Week4_lesson1 (這裡的預測比較不一樣了，(w[:-1],[-1]) -> (w[:-1],[1:]),用conv1d和lstm建model):
https://github.com/hao134/Tf_course/blob/main/Tf_course4/Week4_lesson1.py
* 7.Week4_lesson3 (真實數據sunspots 用 (w[:-1],[-1])的方式預測）
https://github.com/hao134/Tf_course/blob/main/Tf_course4/Week4_lesson3.py
* 8.Week4_lesson5 (真實數據sunspots 用 (w[:-1],[1:])的方式預測，用Conv1D,lstm模型預測）
https://github.com/hao134/Tf_course/blob/main/Tf_course4/Week4_lesson5.py

* Week2'e Exercise (在簡單的時序數據，用調整後的learning rate 和簡單的dnn模型，使mae降低）：
https://github.com/hao134/Tf_course/blob/main/Tf_course4/Week2_exercise.py
* Week3's Exercise (跟上面相同，但用lstm model):
https://github.com/hao134/Tf_course/blob/main/Tf_course4/Week3_exercise.py
* Week4's Exercise (真實溫度數據，利用和sunspot類似的方式處理)：
https://github.com/hao134/Tf_course/blob/main/Tf_course4/Week4_exercise.py