# Agreement/Disagreement LSTM model on dialogue data

In this project we will be replicating a 2019 paper by Mikael Apel, Marianna Blix Grimaldi and Isaiah Hull from Sveriges Riksbank (Central Bank of Sweden), titled, "How Much Information Do Monetary Policy Committees Disclose? Evidence from FOMC's Minutes and Transcripts." This paper is in the field of central bank communication, monetary policy and machine learning. This paper does not have any scripts attached to it, so we will be intepreting the methedology from the paper to recreate their results.

<img src="images/img_1.png">

We will investigate a specific part of this paper; the section measuring agreement and disagreement in Federal Open Market Committee meeting transcripts. In the paper, they measure agreement by performing deep transfer learning, a technique that involves training a deep learning model on one set of documents - U.S. congressional debates - and then making predictions on another: FOMC meeting transcripts. This is because of FOMC meeting transcripts are unlabeled. Overall, the paper finds that transcripts are more informative than minutes and heightened committee agreement typically preceds policy rate increases.

The deep learning model to predict agreement using U.S. congressional debate corpus that contained a vote(yes or no) label that is sufficiently large. This corpus is an ideal choice because it associates speech text with a vote that indicates whether a speaker is agreeing or disagreeing with a bill. After training the deep learning model to achieve high out-of-sample prediction accuracy, we then use it to classify text from FOMC transcripts, thus giving us a novel measure of committee agreement. (Apel 2019)

The goal of this project is to recreate the Apel 2019 paper's Figure 5, shown below. 
<img src="images/img_2.png">

## Model Details

We base our model architecture on the details outlined in the Apel 2019 paper. The paper employs a "transductive" form of transfer learning. More specefically the paper uses a neural network with a long short term memory (LSTM) architecture and word embeddings. The deep learning model takes a sequence of word vectors and converts them into integer indices. The embeddings are then connected to a lyer of long short term memory cells. We then flatten the output from the LSTM cells in a vector and connect them to the output layer via a dense layer. The model uses a binary crossentropy loss function and encode agree as 1 and disagree as 0. The model uses rectified linear activation fuctions on all the hidden layers and apply 25% dropout to the final dense layer. Using the adam optimizer, we train the model on 80% ofthe training sample and 20% on the validation sample. The final model accuracies from the paper are 68% in-sample and 65% out-of-sample. The paper notes that these values are derive from the classification of all statements, whether or not they contain signs of agreement; in many cases the model is roghly ndifferent for such statements and generates a probability of agreement near 0.5.

After training of the U.S. congressional debate data, we use the model to generate probabilities of agreement for each FOMC member's statements in the meeting transcripts. Then we average over the statements to compute a mean probability for the entire meeting transcript. 

Here is an example of the training dataset from the U.S. congressional debate dataset.

***Train Set:*** *I thank the gentleman for yielding me this time and for his great work on this bill. Mr. chairman, this country needs to create a new energy landscape that begins shrinking our disproportionate reliance on foreign energy sources and begins building one that places American ingenuity, producers and consumers at the forefront. I want to highlight one provision and that is the provision that signi cantly strengthens the important leaking underground storage tank program. The bill increases state funding... I urge their support and the support of the bill.*

Here is an example from the test dataset from the FOMC meeting transcripts. 

***Test Set:*** *Based on research from my staff, I have also lowered my estimate of ... real gdp growth ... reflecting expectations of trend growth for both the labor force and labor productivity.*



