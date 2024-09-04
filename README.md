<h2 align= "center"><em>Facebook Friend Recommendation Using Graph Mining</em></h2>

<div align="center">
  <img height="400" src="https://github.com/shreyjain99/TagGen-Automated-Tagging-for-Stack-Overflow-Questions/blob/main/src%20files/pic1.jpg"/>
</div>

<hr width="100%" size="2">

<h3 align= "left"> <b> Key Project Formulation </b> </h3>

<br>

<p>
<strong>Real World/Business Objective :</strong> Given a directed social graph, have to predict missing links to recommend users
</p>

<br>

<p>
<strong>Constraints :</strong>
</p>
<ol>
<li>Probability of prediction is useful to recommend ighest probability links</li>
<li>No low-latency requirement.</li>
</ol>

<br>

<p>
<strong>Get the data from :</strong> https://www.kaggle.com/c/FacebookRecruiting
<br>The data is from facebook recruiting prediction competition hosted on kaggle.
</p>

<br>

<p>
<strong>Data Collection :</strong>
<br>
</p>
<p>Taken data from facebook's recruting challenge on kaggle https://www.kaggle.com/c/FacebookRecruiting
data contains two columns source and destination eac edge in graph - Data columns (total 2 columns):
- source_node int64
- destination_node int64</p>

<br>

<p>
<strong>ML Problem Formulation :</strong>
</p>
<p> <strong>Link Prediction in graph</strong> </p>
<p> 
Generated training samples of good and bad links from given directed graph and for each link got some features like no of followers, is he followed back, page rank, katz score, adar index, some svd fetures of adj matrix, some weight features etc. and trained ml model based on these features to predict link.
</p>

<br>
<br>

<p>
<strong>Performance metrics :</strong>
</p>
<ol>
<li>Both precision and recall is important so F1 score is good choice </li>
<li>Confusion matrix</li>
</ol>

<hr width="100%" size="2">

<br>

<body>

  <h3>Flow of Project : </h3>
  
  <br>

  <h3 align= "center"><strong>Data Loading</strong></h3>

  <div align= "center">|</div>
  <div align= "center">|</div>
  <div align= "center">\/</div>

  <h3 align= "center"><strong>Analysis of tags</strong></h3>
  <p align= "center"><em> - Total number of unique tags </em></p>
  <p align= "center"><em> - Number of times a tag appeared </em></p>
  <p align= "center"><em> - Tags Per Question </em></p>
  <p align= "center"><em> - Most Frequent Tags </em></p>
  <p align= "center"><em> - The top 20 tags </em></p>

  <div align= "center">|</div>
  <div align= "center">|</div>
  <div align= "center">\/</div>

  <h3 align= "center">Data Preprocessing (Natural Language Processing)</h3>
  <p align= "center"><em> - Separated out code-snippets from Body </em></p>
  <p align= "center"><em> - Removed Special characters from Question title and description </em></p>
  <p align= "center"><em> - Removed stop words (Except 'C') </em></p>
  <p align= "center"><em> - Converted all the characters into small letters </em></p>
  <p align= "center"><em> - Used SnowballStemmer to stem the words </em></p>

  <div align= "center">|</div>
  <div align= "center">|</div>
  <div align= "center">\/</div>

  <h3 align= "center">Machine Learning Models</h3>
  <p align= "center"><em> - Converted tags for multilabel problems </em></p>
  <p align= "center"><em>- Splited the data into test and train (80:20)  </em></p>
  <p align= "center"><em> - Featurized data (TFIDF FEATURES) </em></p>
  <p align= "center"><em> - Applyied Logistic Regression with OneVsRest Classifier</em></p>


  
</body>

<hr width="100%" size="2">
<br>

<div align="center">
  <img height="400" src="https://github.com/shreyjain99/TagGen-Automated-Tagging-for-Stack-Overflow-Questions/blob/main/src%20files/frequent%20tags.png"/>
</div>

<p>
<strong>Future Scope :</strong>
</p>
<ol>
<li>Use bag of words upto 4 grams and compute the micro f1 score with Logistic regression(OnevsRest) </li>
<li>Perform hyperparameter tuning on alpha (or lambda) for Logistic regression to improve the performance using GridSearch</li>
<li>Try OneVsRestClassifier with Linear-SVM (SGDClassifier with loss-hinge)</li>
</ol>

<hr width="100%" size="2">
