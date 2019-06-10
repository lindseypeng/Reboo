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
		<a href="http://platoni.city/">Slides</a>
		•
		<a href="http://platoni.city/">Docker</a>
	</strong>
</p>


## Problem Statement

Book market is billion dollar business A problem is that growth rate of book published is much faster than growth rate of sales. This phenomenon has led to millions of dollars poured into competing for people's attentions, which is why a smart automated intelligent recommendation system can acts as a filter, delivers the right content to the right people. However current recommendation is limited with two main problem:

- echo chamber, users find book too boring 
- seemingly similar genre but completely irrelevant for users

## Solution
Using mapping ideas in the actua ideas of the books, I aim to re-classify ideas in a different vector space and brings books that might seem different but actually similar together. Using Natural language modelling, books that were never exposed can be recommended to users with explanations. The long term goal is to 

- reduce $ sunk cost of content creation
- reduce cost of zero sum game of competition.

## Overview

* **Base Case.** Recommendation from nearest neighbor of a 2D representation of your book description. Description vector was reduced with t-sNE(dim=2) from Bert-embeddings(dim=768). 
* **MultiLabel Classification** Each book description is embedded in a dim=6 vector corresponding to genres. 
* **Web App** Book recommendations based on the description of your book. It tells you why the book was recommended by giving you the nearest neighbor in 2D t-sNE representation.
* **Docker** Test the scripts anywhere you are without worrying about dependencies. 

-For more information, visit slide deck:  

---

## Table of Contents 


- [Docker](#docker)
- [Example](#example)
- [Web_App](#web_app)
- [Slides](#slides)
- [License](#license)


## Docker

- All the `code` required to get started
- Images of what it should look like

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
### Web_App

- If you want more syntax highlighting, format your code like this:

![Recordit GIF](https://raw.githubusercontent.com/lindseypeng/Reboo/master/pics/webapp.gif)

---

## Slides

***INSERT ANOTHER GRAPHIC HERE***

[![INSERT YOUR GRAPHIC HERE](http://i.imgur.com/dt8AUb6.png)]()

- Most people will glance at your `README`, *maybe* star it, and leave
- Ergo, people should understand instantly what your project is about based on your repo

> Tips

- HAVE WHITE SPACE
- MAKE IT PRETTY
- GIFS ARE REALLY COOL

> GIF Tools

- Use <a href="http://recordit.co/" target="_blank">**Recordit**</a> to create quicks screencasts of your desktop and export them as `GIF`s.
- For terminal sessions, there's <a href="https://github.com/chjj/ttystudio" target="_blank">**ttystudio**</a> which also supports exporting `GIF`s.

**Recordit**


**ttystudio**

![ttystudio GIF](https://raw.githubusercontent.com/chjj/ttystudio/master/img/example.gif)


> update and install this package first

```shell
$ brew update
$ brew install fvcproductions
```

> now install npm and bower packages

```shell
$ npm install
$ bower install
```

- For all the possible languages that support syntax highlithing on GitHub (which is basically all of them), refer <a href="https://github.com/github/linguist/blob/master/lib/linguist/languages.yml" target="_blank">here</a>.
---

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright 2015 © <a href="http://fvcproductions.com" target="_blank">FVCproductions</a>.
