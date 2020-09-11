# ADL HW2
python = 3.7

1. Model訓練<br>
	```
	# the path of the train file = '../data/train.json'
	# the path of the dev file = '../data/dev.json'
	python src/train.py
	```
	進行模型訓練並於每一個epoch儲存相對應的model_state，最後採納epoch = 2的model_state當最後的測試模型。<br>

2. Model測試<br>
	```
	python src/eval.py /path/to/test.json /path/to/predict.json /path/to/model_state
	```
	經模型預測出來的答案會存到predict.json。
	```
	python ./scripts/evaluate.py /path/to/test.json /path/to/predict.json /path/to/result.json /path/to/ckip_model/
	```
	產生overall、answerable及unanswerable的F1和EM並存到result.json。<br>
3. run.sh內部說明<br><br>
	```
	bash run.sh /path/to/text.json /path/to/predict.json
	```
	下面為內容

	```
	TEST_INPUT_PATH="${1}"
	PREDICT_OUPUT_PATH="${2}"
	MODEL_PATH="model_state/ckpt.2.pkl"
	python3.7 src/eval.py $TEST_INPUT_PATH $PREDICT_OUPUT_PATH $MODEL_PATH
	```
		
3. 資料夾說明<br>
	```
	r08922194
	|___model_state #每個模型各自的model_state	
		|__ckpt.2.pkl #bash download.sh會產生
	|___src		
		|___xxxx.py #所有模型會用到的程式碼和畫圖的程式碼
	|___scripts
		|___evaluate.py #計算分數
	|___requirements.txt
	|___Report.pdf
	|___README.md
	|___run.sh
	|___early.sh
	|___download.sh
	|___threshold 	#畫threshold.png會用到
		|__predict
			|__0.1_predict.json
			|__0.3_predict.json
			|__0.5_predict.json
			|__0.7_predict.json
			|__0.9_predict.json
		|__result
			|__0.1_result.json
			|__0.3_result.json
			|__0.5_result.json
			|__0.7_result.json
			|__0.9_result.json
	|___distribution.png
	|___threshold.png
	```
	<br>

4. Q5：Answer Length Distribution<br><br>

	```
	python src/ans_len_distribution.py /path/to/train.json
	```
		
5. Q6：Answerable Threshold<br><br>

	```
	python src/create_threshold_json.py path/to/dev.json ./model_state/ckpt.2.pkl
	```
	產生五個threshold的predict.json，並存放在./threshold/predict裡。
	```
	python ./scripts/evaluate.py /path/to/dev.json ./threshold/predict/0.1_predict.json ./threshold/result/0.1_result.json /path/to/ckip_model/
	python ./scripts/evaluate.py /path/to/dev.json ./threshold/predict/0.3_predict.json ./threshold/result/0.3_result.json /path/to/ckip_model/
	python ./scripts/evaluate.py /path/to/dev.json ./threshold/predict/0.5_predict.json ./threshold/result/0.5_result.json /path/to/ckip_model/
	python ./scripts/evaluate.py /path/to/dev.json ./threshold/predict/0.7_predict.json ./threshold/result/0.7_result.json /path/to/ckip_model/
	python ./scripts/evaluate.py /path/to/dev.json ./threshold/predict/0.9_predict.json ./threshold/result/0.9_result.json /path/to/ckip_model/
	```
	產生五個threshold的result.json，並存放在./threshold/result裡。
	```
	python src/plot_threshold.py
	```
	讀取./threshold/result資料夾裡不同threshold產生的F1和EM並繪製成圖。












