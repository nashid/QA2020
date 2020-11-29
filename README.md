## Learning to Recommend Relevant API Documentation for Answering Developer Questions

### Requirements

```
pip install numpy scipy gensim sklearn tensorflow==1.13.1 keras==2.2.4 nltk
```

### Prepare your data

To train the model, you need to put your data into the Repository root path. 

```
data_type/
    Doc_list.txt
    QA_list.txt
```

1. `Doc_list.txt` contains all the API Documents, one document per line.
2. `QA_list.txt` contains Question-Documents pairs. Each pair has the Question and the indexes of relevant API documents. The indexes are the line number of the `Doc_list.txt` file, start from 1. An example of the `QA_list.txt`:

```
How to display all my items? ...
307  556  557  558  559
How to programmatically access ...
219  466  467  234  474  475  232  473
...
``` 

### Training and evaluation

Modify the parameters in the `run.sh` and run it:

`sh run.sh`

