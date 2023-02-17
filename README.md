
## Dimitry Kirtsman Data Science Projects

### Forecasting Stock Prices Using LSTM and Transformer Architectures
The paper investigated the efficacy of LSTM and Transformer models in forecasting one-day ahead stock price returns. In addition, both models used the Time2Vec algorithm as a positional embedding layer. The stocks considered in the analysis were Amazon (AMZN), Intel (INTC), Pfizer (PFE), Proctor & Gamble (PG), and Verizon (VZ). The models used a ten-year time series from October 1, 2012, to September 30, 2022, with price, volume, and a variety of technical indicators serving as the features for each model. Training for all models used a cross-validation design with hyperparameter tuning. RMSE and MAPE were utilized to evaluate price level forecast performance. In addition, binary (up/down) signals were created from the price forecasts and assessed using accuracy and F1-scores. Study results showed no significant performance differences between the two models.

**Software:** Python, Keras/Tensorflow

**Methods:** LSTMs, Transformers, Time2Vec, Time series, Cross-validation and Hyperparameter Tunning. 

[Link to Paper](https://drive.google.com/file/d/1AqRlX8aUwSOF8vcj7Sj1nF6uQ17JUnL0/view?usp=sharing) 

[Link to Code](https://github.com/dimitryk77/Stock_Models/tree/main/Model%20Code)  

<br />


### Classification of Cyberbullying Tweets Using LTSM, GRU, CNN and Transformer Models
The paper explores the efficacy of LSTM, GRU, CNN, and Transformer models for text classification of cyberbullying tweets. The analysis consists of 8 experiments (12 models in all) and utilizes a dataset of more than 47,000 cyberbullying tweets and the Keras library in Python. The experiments start with data cleaning and EDA. The next five experiments utilize cross-validation and hyperparameter tuning to test various structures of LSTMs, GRUs, their bidirectional counterparts, and CNNs. These experiments utilize trained embedding and pre-trained tweet-based GloVe embeddings. The last two experiments involve an attention-based transformer and a pre-trained BERT model. The best model from the analysis is a hyperparameter-tuned attention-based transformer, which obtained an 88.04% accuracy on the test set.

**Software:** Python, Keras/Tensorflow

**Methods:** LSTMs, GRUs, CNNs, Transformers, GloVe Embeddings, Cross-validation and Hyperparameter Tunning.

[Link to Paper](https://drive.google.com/file/d/1Ft7iTyTg9Jyotnvd1fbVv2C8xKdDwdIM/view?usp=sharing) 

[Link to Code](https://github.com/dimitryk77/Cyberbullying-Tweets-Models/tree/main/Model_Code)  

<br />

### Image Classification Using ANN, CNN, and ResNet50 Neural Network Architectures
The paper explores the selection of a proper deep neural network architecture, ANN, CNN, or the ResNet50 transfer learning model for image classification. The analysis consists of 7 experiments (12 models in all), utilizing the CIFAR-10 dataset and the Keras library in Python. The experiments start with the analysis of ANNs and CNNs with two and then three hidden layers (convolution/max-pooling layers for CNNs). Next, regularization is introduced for all of the models in the form of dropout and batch normalization. Additional experiments are conducted utilizing hyperparameter tuning to improve performance and data augmentation to further decrease overfitting. The final experiment explores transfer learning using the ResNet50 CNN architecture, which achieved the best CIFAR-10 test set accuracy score of all the models at 86.18%.

**Software:** Python, Keras/Tensorflow

**Methods:** ANNs, CNNs, ResNet50, Transfer Learning, Data Augmentation, Cross-validation and Hyperparameter Tunning. 

[Link to Paper](https://drive.google.com/file/d/1LUuux5frpF5OSHiTokXstBN2hpwAPiC0/view?usp=sharing) 

[Link to Code](https://github.com/dimitryk77/Image-Classification-Models/tree/main/Model%20Code) 

<br />

### DBSCAN, Isolation Forest, and Autoencoders for Anomlay Detection of Sales Transactions Data
The paper explores the efficacy of DBSCAN, Isolation Forest, and Autoencoders in detecting fraudulent sales transactions. The dataset contains variables related to product sales and is comprised of 133,731 train set observations and 15,732 test set observations. Data cleansing and feature creation are first applied to both data sets. Each of the models is then fit on the train set, and predictions of fraud/not fraud are obtained from the test set. Different iterations for each of the models above are tested by varying model parameters. Model performance is then evaluated by comparing the model’s test set predictions of fraud/not fraud versus the actual fraud outcomes using the F1 score.

**Software:** R

**Methods:** DBSCAN, Isolation Forest, Autoencoders, Anomaly Detection. 

[Link to Paper](https://drive.google.com/file/d/1RXVmnnzO-ZwQj5ObKS6x2-dZ5G-FoQhX/view?usp=sharing) 

[Link to Code](https://github.com/dimitryk77/Image-Classification-Models/tree/main/Model%20Code) 

<br />
