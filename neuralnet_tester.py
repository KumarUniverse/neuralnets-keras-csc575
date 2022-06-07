from wine import *


def eval():
    model = train_model()

    # load the testing dataset
    dataset = loadtxt('wine_test.csv', delimiter=',')
    print(type(dataset))

    # split into input (X) and output (y) variables
    X = dataset[:, 1:]
    y = dataset[:, 0]
    y = to_categorical(y)

    # PREDICTIONS & EVALUATION

    # Final evaluation of the model
    scores = model.evaluate(X, y, verbose=0)
    print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

    return scores[1]

def test_60_percent():
    assert eval() >= 0.6

def test_70_percent():
    assert eval() >= 0.7

def test_80_percent():
    assert eval() >= 0.8

def test_90_percent():
    assert eval() >= 0.9

