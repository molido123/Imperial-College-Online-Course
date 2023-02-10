# Tutorial 1: Computing an inter-annotation agreement for a new dataset

**Note:** Python 3.x is required to run this code. All files can be run independently.

This exercise contains 1 python file to convert the original json file into csv, plus 2 other exercise files.

**iaa_simple.py** showcases the various annotation metrics computed on a new dataset, for the simplified binary case.
The students are required to provide their own implementation for the metrics based on the formula presented in the
course.

**iaa_extended.py** extends the binary case to multiple labels. Students are required to implement a few methods and
display the confusion matrix.

## Libraries

- pandas: ```pip install pandas```
- sklearn: ```pip install sklearn```

## Dataset

The dataset has been downloaded from https://www.kaggle.com/rtatman/discourse-acts-on-reddit. Please see license for
more details. 