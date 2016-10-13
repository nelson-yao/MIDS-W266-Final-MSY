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
