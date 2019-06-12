<h1 align="center">
	<img
		width="300"
		alt="The Lounge"
		src="https://scontent-yyz1-1.xx.fbcdn.net/v/t1.0-9/62200292_10157411681371974_7461257159871823872_n.jpg?_nc_cat=107&_nc_ht=scontent-yyz1-1.xx&oh=c76009347bb601ab5d8cdd894ce0467c&oe=5D80E6C7">
</h1>

<h3 align="center">
	ReClassify Ideas For Book Recommendations
</h3>

<p align="center">
	<strong>
		<a href="http://platoni.city/">Website</a>
		•
		<a href="https://bit.ly/2I8fdfK">Slides</a>
		•
		<a href="https://cloud.docker.com/u/alinsi/repository/docker/alinsi/reboo">Docker</a>
	</strong>
</p>

## Table of Contents 

---
- [Motivation](#motivation) 
- [Solution](#solution) 
- [Pipeline](#pipeline)
- [Docker](#docker)
- [Example](#example)
- [Demo](#demo)
---

## Motivation

Book market is billion dollar business. A problem is that growth rate book publishing is much faster than growth rate of sales. This phenomenon has led to millions of dollars poured into competing for people's attentions, which is why a smart automated intelligent recommendation system is important in delivering the right content to the right people. 

However current recommendation is limited:

- echo chamber, users find book that are too similar too boring 
- seemingly similar genre but completely irrelevant for users

## Solution
By mapping ideas in the actual ideas of the books, Reboo makes conten-based book recommendation system using natural language processing. Book which might seem uncorrelated are actually related if you parse through the contents.  In addition, Reboo can provide users explanations(see more explanation in pipeline) why the books were recommended from the database. 

<h1 align="center">
	<img
		src="https://raw.githubusercontent.com/lindseypeng/Reboo/master/pics/rb2.png">
</h1>


## Pipeline
The naive approach is to translate english description of books into vector representations using word embedding, reducing the vector to 2D plot and using closest neighbor in 2D plot to make recommendation.

However, to scale it up, we need a way to finetune and train the model to target books into specific clusters. This require us to train the model to crate new-genres when we don't know whats the new topic connecting ideas without having someone going through each single book.

Reboo approximate a new genre by creating a new inter-genre vector that represents a vector of different genres tagged by readers. Currently, a book can be simultaneously up to six different genres. Reboo  train a deep RNN(BERT) to classify book description into a multilabel genres and the vector is further reduced into 2D plot and new clusters are estimated to be the new genres. This process requires further iterations and more data. 

<h1 align="center">
	<img
		src="https://raw.githubusercontent.com/lindseypeng/Reboo/master/pics/pipeline.png">
</h1>

## Overview

* **Base Case.** Content-Based recommendation using word embedding and closest neighbor. Section in example show how to access this functionality. The input is book description and output is a list of book title and the sentence that was mapped together with your description. You can change how many neighbors (Recommendations) you want. 
* **MultiLabel Classification** Each book description is embedded in a dim=6 vector corresponding to genres. The input is 
book description and the output is a vector of 6 for each sentence of your book. The script is in the file MULTI-LABEL-CLASSIFICATIONBERT.ipynb. You can modify the script to process entire book description instead of separate sentences.
* **Web App** Book recommendations based on the description of your book. The backbone currently runs on the Base Case. A demo of this web app is shown in gif in demo section.
* **Docker** A machine learning container including other packages required to run Reboo Scripts. 

---


## Docker

- download and run docker container with all the neccessary dependencies for this repo.
- tags are added to access visualization on local computer.
- tags are added to remove the container after existing.
- when you call the command you are taken into the bash terminal of the container. 

```bash
xhost +
```
```bash
sudo docker run -it --rm --net=host --ipc=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unit -v /home/lindsey/Desktop:/root alinsi/reboo
```
---
## Example 

> import pandas and the main script bert_emb_module, which are included in docker container
```python
import pandas as pd
from bert_emb_modul import mastercode as ms
```
> copy paste a book description of your choice
```python
description="In this landmark book, Scott Page redefines the way we understand ourselves in relation to one another. \
The Difference is about how we think in groups--and how our collective wisdom exceeds the sum of its parts. \
Why can teams of people find better solutions than brilliant individuals working alone? And why are the best group \
decisions and predictions those that draw upon the very qualities that make each of us unique? The answers lie \
in diversity--not what we look like outside, but what we look like within, our distinct tools and abilities."
```
> method 0 is without tfidf and method 1 is with tfidf. Initalize our instance with the description and method
```python
method=0
obj=ms(description,method)
```
> get a list of recommendation in Panda frames using the closest neighbor method. See Cluster method in document.
> choose how many neighbors you want to see in n_neighbors
```python
##get recommendation based o n closest neighbors##
result,avgs,current=obj.master()
n_neighbors=5
suggestion=obj.output_neighbors(n_neighbors)
recommendation=obj.recommend_closest_neighbor(suggestion,result)
Closest_recomm=obj.clean_rec_close_neighbor(recommendation)
```
> from the list of recommendation print out the booktitle and associated sentence that was the closest neighbors
```python
def cleanstrings(list1):
    list2=str(list1).replace('[','').replace(']','')
    list3=list2.replace('"\'','').replace('\'"','')
    return list3

##print out recommendation for closest neighbor iterative
def print_close_recommend(Closest_recomm):
    for j,row in Closest_recomm.iterrows():
        booktitle=str(row['booktitle'])
        booktitle_clean=cleanstrings(booktitle)
        sentences=str(row['book_sentences'])
        sentences_clean=cleanstrings(sentences)
        #print('Title:{} \n {}'.format(booktitle_clean,sentences_clean))

```

---
### Demo


![Recordit GIF](https://raw.githubusercontent.com/lindseypeng/Reboo/master/pics/webapp.gif)

---
## DataSets
- Data is scrapped from GoodReads.com book descriptions.
- Data are stored in Datasets folder

---








