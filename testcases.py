import project2
import pytest

def test_extraction():
    data_1=project2.extraction()
    assert data_1 is not None

def test_extractredact():
    datared=project2.extractredact()
    assert datared is not None

def test_extractnames():
    data_1=project2.extraction()
    y_train=project2.extract_names(data_1)
    assert type(y_train)==list

def test_extractfeatures():
    data_1=project2.extraction()
    y_train=project2.extract_names(data_1)
    x=list(project2.extract_features(y_train))
    assert type(x)==list

def test_test():
    datared=project2.extractredact()
    x_test=list(project2.test(datared))
    assert type(x_test)==list

