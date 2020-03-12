# import the necessary packages
import argparse
import numpy as np
from Softmax.linear_classifier import Softmax
import Softmax.data_utils as du
import torch
from Net import Net
import pickle


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
    
   
    net = Net(n_feature=3072, n_hidden=500, n_output=10)
    
    net.load_state_dict(checkpoint)
    
    X =torch.from_numpy(X)
    X=X.float()
    predicted = net.predict(X).data
    #prediction = checkpoint(X)

    scores=predicted.numpy()
    y_pred = scores.argmax(axis=1)

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
    
    with open('Softmax/softmax_weights.pkl', 'rb') as f:
        W = pickle.load(f)
        
    
    new_softmax = Softmax()
    y_pred = np.zeros(X.shape[0])

    
    scores = np.dot(X,W)
    y_pred = scores.argmax(axis=1)

    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return y_pred

def main(filename, group_number):

    X,Y = du.load_CIFAR_batch(filename)
    print(X.shape, Y.shape)
    X = np.reshape(X, (X.shape[0], -1))
    mean_image = np.mean(X, axis = 0)
    X -= mean_image
    prediction_pytorch = predict_usingPytorch(X)
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    prediction_softmax = predict_usingSoftmax(X)
    acc_softmax = sum(prediction_softmax == Y)/len(X)
    print(prediction_pytorch.shape, Y.size)
    acc_pytorch = sum(prediction_pytorch == Y)/len(X)
    print("Group %s ... Softmax= %f ... Pytorch= %f"%(group_number, acc_softmax, acc_pytorch))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--test", required=True, help="path to test file")
    ap.add_argument("-g", "--group", required=True, help="group number")
    args = vars(ap.parse_args())
    main(args["test"],args["group"])
