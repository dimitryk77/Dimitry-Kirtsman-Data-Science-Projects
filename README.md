
## Dimitry Kirtsman Data Science Projects

### Forecasting Stock Prices Using LSTM and Transformer Architectures
The paper investigated the efficacy of LSTM and Transformer models in forecasting one-day ahead stock price returns. In addition, both models used the Time2Vec algorithm as a positional embedding layer. The stocks considered in the analysis were Amazon (AMZN), Intel (INTC), Pfizer (PFE), Proctor & Gamble (PG), and Verizon (VZ). The models used a ten-year time series from October 1, 2012, to September 30, 2022, with price, volume, and a variety of technical indicators serving as the features for each model. Training for all models used a cross-validation design with hyperparameter tuning. RMSE and MAPE were utilized to evaluate price level forecast performance. In addition, binary (up/down) signals were created from the price forecasts and assessed using accuracy and F1-scores. Study results showed no significant performance differences between the two models.

[Link to Paper](https://drive.google.com/file/d/1AqRlX8aUwSOF8vcj7Sj1nF6uQ17JUnL0/view?usp=sharing) 

[Link to Code](https://github.com/dimitryk77/Stock_Models/tree/main/Model%20Code)  

<br />


### Classification of Cyberbullying Tweets Using LTSM, GRU, CNN and Transformer Models
The paper explores the efficacy of LSTMs, GRUs, CNNs, and Transformer models for text classification of cyberbullying tweets. The analysis consists of 8 experiments (12 models in all) and utilizes a dataset of more than 47,000 cyberbullying tweets and the Keras library in Python. The experiments start with data cleaning and EDA. The next five experiments utilize hyperparameter tuning to test various structures of LSTMs, GRUs, their bidirectional counterparts, and CNNs. These experiments utilize trained embedding and pre-trained tweet-based GloVe embeddings. The last two experiments involve an attention-based transformer and a pre-trained BERT model. The best model from the analysis is a hyperparameter-tuned attention-based transformer, which obtained an 88.04% accuracy on the test set.

[Link to Paper](https://drive.google.com/file/d/1AqRlX8aUwSOF8vcj7Sj1nF6uQ17JUnL0/view?usp=sharing) 

[Link to Code](https://github.com/dimitryk77/Cyberbullying-Tweets-Models/tree/main/Model_Code)  

<br />

