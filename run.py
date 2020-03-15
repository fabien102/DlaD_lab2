# import the necessary packages
import Softmax.data_utils as du
import argparse
import numpy as np
import torch
from Softmax.linear_classifier import Softmax
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
    
    
    X = np.reshape(X, (X.shape[0], -1))

    checkpoint = torch.load("Pytorch/model.ckpt",map_location=lambda storage, loc: storage)
    
    
   
    net = Net(n_feature=3072, n_hidden=500, n_output=10)
    
    net.load_state_dict(checkpoint)
    
    predicted = net.predict(X).data
    
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
    ### Modified this part
    mean_pytorch = np.array([0.4914, 0.4822, 0.4465])
    std_pytorch = np.array([0.2023, 0.1994, 0.2010])
    X_pytorch = np.divide(np.subtract( X , mean_pytorch[np.newaxis,np.newaxis,:]), std_pytorch[np.newaxis,np.newaxis,:])
    prediction_pytorch = predict_usingPytorch(torch.Tensor(np.moveaxis(X_pytorch,-1,1)))
    ####
    X = np.reshape(X, (X.shape[0], -1))
    mean_image = np.mean(X, axis = 0)
    X -= mean_image
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



