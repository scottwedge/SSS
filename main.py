immport tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)
#integers  encoded words, basically pointers to a word bank so print(train_data[0]) doesnt work

word_index = data.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}

word_index["<PAD>"] = 0
word_index["<START"] = 1
word_index["UNK"] = 2
word_index["<UNUSED>"] = 3

#swap so that integer points to word. kinda dumb
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=256)


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

#print(decode_review(test_data[0]))
#print(len(test_data[0]), len(test_data[1]))

model = keras.Sequential()

#Overall: input sentence -> word vocab -> get 16 vector values -> average layer, which averages each n-vector for all of the words in the review
# -> maps the average value for the 16 vectors into 16 neurons (relu) -> dense layer(relu) where you learn->output value using sigmoid (0-1)

#embedding layer
#create vectors for each word that has 16 dimensions (Ax +by +cz + dw.....
#when we learn each vector, we assign different values to the variables of the vector such that similar words have "less of a difference" between each other
#we then take the average of each dimension for each word, then pass them into 16 neurons
model.add(keras.layers.Embedding(88000, 16))

#This layer scales down 16 dimension to a lower dimension since 16 is alot for us to compute
model.add(keras.layers.GlobalAveragePooling1D())

#dense layer here
#this part will check the word around and learn if the review is good or not
#then adjust biases and weights to feed into sigmoid
model.add(keras.layers.Dense(16, activation="relu"))

#changes answer to either zero for bad or one for good
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()
#output is zero or one, loss function calculates loss between 1-0 and the actual value (lets say 0.2)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)

model.save("model.h5")


#if running it multiple times, comment above and use the command below
#model = keras.models.load_model("model.h5")

def review_encode(s):
	encoded = [1]

	for word in s:
		if word.lower() in word_index:
			encoded.append(word_index[word.lower()])
		else:
			encoded.append(2)

	return encoded
	
with open("C:/Users/aidan/Dev/ML/Test/test4.txt", encoding="utf-8") as f:
	for line in f.readlines():
		nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
		encode = review_encode(nline)
		encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
		predict = model.predict(encode)
		print(line)
		print(encode)
		print(predict[0])

'''
test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(test_labels[0]))
'''
