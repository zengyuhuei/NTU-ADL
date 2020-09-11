# ADL HW1
使用老師提供的前處理sample code<br>
python = 3.7
1. 建立符合作業需求的環境以及下載glove.840B.300d.txt<br>
	```
	bash install_packages.sh
	download glove.840B.300d.txt from https://nlp.stanford.edu/projects/glove/
	```
2. 資料前處理<br>
	```
	python src/preprocess_seq_tag.py datasets/seq_tag/
	python src/preprocess_seq2seq.py datasets/seq2seq/
	python src/preprocess_seq2seq.py datasets/attention/
	```
	產生訓練要使用的train.pkl、valid.pkl及embedding.pkl，此外，因為test.pkl需要使用已經存在的embedding.pkl，所以必須使用下列指令產生：
	```
	python src/preprocess_seq_tag_test.py /path/to/test.jsonl /path/to/test.pkl  /path/to/embedding
	python src/preprocess_seq2seq_test.py /path/to/test.jsonl /path/to/test.pkl  /path/to/embedding
	```
	(此程式是根據老師給予的前處理做修改)。<br>

3. Seq_tag模型建立<br>
	```
	python src/train_seq_tag.py
	```
	進行模型訓練並於每一個epoch儲存相對應的model_state，最後採納epoch = 10的model_state當最後的測試模型。<br>

4. Seq_tag測試<br>
	```
	python src/eval_seq_tag.py /path/to/test.pkl /path/to/predict.jsonl /path/to/embedding /path/to/model_state
	```
	將valid.pkl當作測試資料進行預測，模型預測出來的結果為1(包含此token)及0(不包含此token)，並利用postprocess選擇文章中句子token為1的數量最多的當作summary，並寫入predict_seq_tag.jsonl，最後利用:
	```
	python scripts/score_extractive.py /path/to/predict.jsonl /path/to/test.jsonl
	```
	產生Rouge-1、Rouge-2及Rouge-L的分數。 <br>

5. Seq2seq模型建立<br>
	```
	python src/train_seq2seq.py
	```
	進行模型訓練並於每一個epoch儲存相對應的model_state，最後採納epoch = 6的model_state當最後的測試模型。此模型包含class RNNEncoder及class RNNDecoder，前者會將文章壓縮成一個context vector並將其隱藏層當作decoder的輸入去做預測，而在此利用了class Seq2Seq去結合兩個model做訓練。<br>

6. Seq2seq測試<br>
	```
	python src/eval_seq2seq.py /path/to/test.pkl /path/to/predict.jsonl /path/to/embedding /path/to/model_state
	```
	將valid.pkl當作測試資料進行預測，模型預測出來的結果經過softmax產生每個字的機率，並選擇機率最高的字組成summary，寫入predict_seq2seq.jsonl，最後利用:
	```
	python scripts/scorer_abtractive.py /path/to/predict.jsonl /path/to/test.jsonl
	```
	產生Rouge-1、Rouge-2及Rouge-L的分數。<br>

7. Attention模型建立<br>
	```
	python src/train_attention.py
	```
	去進行模型訓練並於每一個epoch儲存相對應的model_state，最後採納epoch = 20的model_state當最後的測試模型。此模型只是在前面的seq2seq model多加了attention機制，讓decoder更能專注於他要預測的字。<br>

8. Attention測試<br>
	```
	python src/eval_attention.py /path/to/test.pkl /path/to/predict.jsonl /path/to/embedding /path/to/model_state
	```
	將valid.pkl當作測試資料進行預測，模型預測出來的結果經過softmax產生每個字的機率，並選擇機率最高的字組成summary，寫入predict_attention.jsonl，最後利用:
	```
	python scripts/scorer_abtractive.py /path/to/predict.jsonl /path/to/test.jsonl
	```
	產生Rouge-1、Rouge-2及Rouge-L的分數。<br>

9. 資料夾說明<br>
	```
	r08922194
	|___src
		|___model_state #每個模型各自的model_state
					|___seq_tag
							|__ckpt.10.pt #bash download.sh會產生
					|___seq2seq
							|__ckpt.6.pt #bash download.sh會產生
					|___attention
							|__ckpt.20.pt #bash download.sh會產生
			|___xxxx.py #所有模型會用到的程式碼和畫圖的程式碼
	|___scripts
			|___scorer_abstractive.py #計算分數
			|___score_extractive.py #計算分數
	|___datasets
			|___seq_tag
				|___embedding.pkl #bash download.sh會產生
				|___test.pkl # bash extractive.sh會產生
			|___seq2seq
				|___embedding.pkl #bash download.sh會產生
				|___test.pkl # bash seq2seq.sh會產生 
			|___attention
				|___embedding.pkl #bash download.sh會產生
				|___test.pkl # bash attention.sh會產生
	|___requirements.txt
	|___README.md
	|___install_packages.sh
	|___extractive.sh
	|___seq2seq.sh
	|___attention.sh
	|___download.sh
	|___glove.840B.300d.txt # download from https://nlp.stanford.edu/projects/glove/
	|___attention_weight.png
	|___relative_location.png
	```
	<br>
10.  extractive.sh內部說明<br><br>
		```
		bash extractive.sh /path/to/text.jsonl /path/to/predict.jsonl
		```
		下面為內容

		```
		TEST_INPUT_PATH="${1}"   #/path/to/test.jsonl
		PREDICT_OUPUT_PATH="${2}" #/path/to/predict.jsonl

		TEST_OUPUT_PATH="datasets/seq_tag/test.pkl"
		EMBEDDING_FILE_PATH="datasets/seq_tag/embedding.pkl"
		MODEL_PATH="src/model_state/seq_tag/ckpt.10.pt"

		#test資料前處理，並產生datasets/seq_tag/test.pkl
		python3.7 src/preprocess_seq_tag_test.py $TEST_INPUT_PATH \
										$TEST_OUPUT_PATH \
										$EMBEDDING_FILE_PATH
		# evaluation
		python3.7 src/eval_seq_tag.py $TEST_OUPUT_PATH $PREDICT_OUPUT_PATH $EMBEDDING_FILE_PATH $MODEL_PATH 
		```
		而seq2seq.sh及attention.sh架構相似。
11.  Q4：The distribution of relative locations<br><br>

		```
		python src/plot_relative_location.py datasets/seq_tag/test.pkl datasets/seq_tag/embedding.pkl src/model_state/seq_tag/ckpt.10.pt
		```
		將模型預測出來的文章句子，也就是預測的摘要，去除以該篇文章的句子數量得到相對位置，並將所有文章的相對位置做數量上的統計，並且進行正規化：<br>
		```
		density = the number of the relative locations / the number of the sentences
		```
		最後在利用matplotlib進行長調圖的繪製。
12.  Q5：Visualize the attention weights<br><br>

		```
		python3.7 src/plot_attention_weight.py datasets/attention/test.pkl datasets/attention/embedding.pkl src/model_state/attention/ckpt.20.pt
		```
		將decoder中預測每一個字的attention weight都存起來，並利用matshow將文章的tokens、預測摘要的tokens以及相對應的attention畫成圖，存成attention_weight.png。
	











