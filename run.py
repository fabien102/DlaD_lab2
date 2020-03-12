# import the necessary packages
import argparse
import numpy as np
from Softmax.linear_classifier import Softmax
import Softmax.data_utils as du
import torch
from Net import Net

#########################################################################
# TODO:                                                                 #
# This is used to input our test dataset to your model in order to      #
# calculate your accuracy                                               #
# Note: The input to the function is similar to the output of the method#
# "get_CIFAR10_data" found in the notebooks.                            #
#########################################################################

def predict_usingPytorch(X):
    #########################################################################
    # TODO:                                                                 #
    # - Load your saved model                                               #
    # - Do the operation required to get the predictions                    #
    # - Return predictions in a numpy array                                 #
    #########################################################################
    checkpoint = torch.load("Pytorch/model.ckpt")
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    
    if torch.cuda.is_available():
        net = Net(n_feature=3072, n_hidden=500, n_output=10).cuda(device)     # define the network
    else:
        net = Net(n_feature=3072, n_hidden=500, n_output=10)
    
    net.load_state_dict(checkpoint)
    print(X.shape)
    predicted = net.predict(X).data
    #prediction = checkpoint(X)

    y_pred=prediction.np()
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return y_pred

def predict_usingSoftmax(X):
    #########################################################################
    # TODO:                                                                 #
    # - Load your saved model                                               #
    # - Do the operation required to get the predictions                    #
    # - Return predictions in a numpy array                                 #
    #########################################################################
    with open('softmax_weights.pkl', 'rb') as f:
        W = pickle.load(f)
    new_softmax = Softmax()
    
    y_pred = new_softmax.predict(Xl)

    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return y_pred

def main(filename, group_number):

    X,Y = du.load_CIFAR_batch(filename)
    X = np.reshape(X, (X.shape[0], -1))
    mean_image = np.mean(X, axis = 0)
    X -= mean_image
    prediction_pytorch = predict_usingPytorch(X)
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    prediction_softmax = predict_usingSoftmax(X)
    acc_softmax = sum(prediction_softmax == Y)/len(X)
    acc_pytorch = sum(prediction_pytorch == Y)/len(X)
    print("Group %s ... Softmax= %f ... Pytorch= %f"%(group_number, acc_softmax, acc_pytorch))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--test", required=True, help="path to test file")
    ap.add_argument("-g", "--group", required=True, help="group number")
    args = vars(ap.parse_args())
    main(args["test"],args["group"])
