<<<<<<< HEAD
# DATASCI W266: Natural Language Processing

Welcome to NLP class! This is the main repository we'll be using to distribute 
course materials.

### [Week 1: Introduction](week1/)

### [Week 2: Language Modeling I](week2/)

### [Week 3: Clusters and Distributions](week3/)

### [Week 4: Language Modeling II](week4/)

### [Week 5: Units of Meaning](week5/)

### [Week 6: Information Retrieval](week6/)

### [Week 7: Part-of-Speech Tagging I](week7/)

### [Week 8: Part-of-Speech Tagging II](week8/)

=======
# MIDS-W266-Final-MSY
Repo for W266 NLP final project. Team: Max Shen, Will Monge and Nelson Yao


### Environment setup

In order to ensure that all code is transportable we have specified the minimal **virtual environment** necessary to run thep project. These virtual environments can be managed through [conda](http://conda.pydata.org/docs/using/envs.html)(specially recommended) or virtualenv / virtualenvwrapper.  

*Should you choose to not create a virtualenv, install directly on your raw machine by following step 2 for virtualenv instructions.*  


### With `conda`

```bash
$ conda env create -f config/nlp-environment.yml
$ source activate nlp
```

That's it!  
*For more info on using virtual environments with conda see [here](http://conda.pydata.org/docs/using/envs.html)*

### With `virtualenv`  

* Create a virtual env (from within the folder) and activate it:  

```bash
$ virtualenv nlp
$ source nlp/bin/activate
```  

* Install pre-reqs:

```bash
$ pip install -r config/nlp-requirements.txt
```
>>>>>>> 3e2a4e9ff662cf31338e13a038aa2564a0569aaa
