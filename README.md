# Agreement/Disagreement LSTM model on dialogue data

In this project we will be replicating a 2019 paper by Mikael Apel, Marianna Blix Grimaldi and Isaiah Hull from Sveriges Riksbank (Central Bank of Sweden), titled, "How Much Information Do Monetary Policy Committees Disclose? Evidence from FOMC's Minutes and Transcripts." This paper is in the field of central bank communication, monetary policy and machine learning. This paper does not have any scripts attached to it, so we will be intepreting the methedology from the paper to recreate their results.

<img src="images/img_1.png">

We will investigate a specific part of this paper; the section measuring agreement and disagreement in Federal Open Market Committee meeting transcripts. In the paper, they measure agreement by performing deep transfer learning, a technique that involves training a deep learning model on one set of documents - U.S. congressional debates - and then making predictions on another: FOMC meeting transcripts. This is because of FOMC meeting transcripts are unlabeled. Overall, the paper finds that transcripts are more informative than minutes and heightened committee agreement typically preceds policy rate increases.

The deep learning model to predict agreement using U.S. congressional debate corpus that contained a vote(yes or no) label that is sufficiently large. This corpus is an ideal choice because it associates speech text with a vote that indicates whether a speaker is agreeing or disagreeing with a bill. After training the deep learning model to achieve high out-of-sample prediction accuracy, we then use it to classify text from FOMC transcripts, thus giving us a novel measure of committee agreement. (Apel 2019)

The goal of this project is to recreate the Apel 2019 paper's Figure 5, shown below. 
<img src="images/img_2.png">










