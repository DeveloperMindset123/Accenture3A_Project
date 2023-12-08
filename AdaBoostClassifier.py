# Tutortial that is being followed: https://insidelearningmachines.com/adaboost_classification_algorithm/

#remark, this is an implementation of classification algorithm from scratch known as adaboost_classification_algorithm

'''
Motivation: What is adaboost classifier?
- Adaboost stands for adaptive boosting, this algorithm was originally developed in 1997
Some key things to keep in mind regarding this classification algorithm:
1. This algorithm is strictly designed to handle labels with only two unique labels
2. It is important to make the ensemble of sufficieint size to obtain good results
3. Assumptions of weak learner, used to build the ensemble, should be considered. In general, the weak learner should be a very "simple" model
4. Since boosting is a sequential algorithm, it can be slow to train. A long training period can affect the scalabillity of the model
'''

"""
Understanding the difference between a constructor and destructor in python:
- A constructor helps in initialization of an object i.e. it allocates memory to an object. On the other hand, a destructor deletes the created constructor when it is of no use which means it deallocates the memory of an object

- Note: to distinguishment between public and private function is the following. Function that has syntax such as "__display" is a private function whereas function that has syntax such as "display" would be a public function (unlike C/C++/java where we explicitly state whether a function is public or private)

-Note 2: make sure the methods are indented on the right level, otherwise, calling on the functions won't be recognized by the python interpreter
"""

# import the neccessary libraries --> ensure that conda python interpreter is enabled otherwise we will need to install redundant packages from scratch
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.base import clone   

#let's start with the class definition, initialiser and destructor functions
class AdaBoostClassifier(object):  #this is how a class is defined in python
    #initializer --> also known as the constructor
    def __init__(self,    # here we are specifying the parameters of the constructor function --> __init__ serves as the constructor method in python, it initializes an object when an instance of an class is created, self is a reference to the object that is being created
                 weak_learner: Any,  # Any means this parameter can be any atype (boolean, float, int, etc)
                 n_elements: int = 100,   # specifies a parameter named n_elements with a default value of 100 and a type hint 'int'
                 record_training_metrics: bool = False) -> None:   # specifies a parameter named record_training_metrics with a default value of False and a type hint bool, None indicates that the method returns None (think of a void function in C/C++/Java)
        # set the values of the parameter to the object instance that is created
        self.weak_learner = weak_learner  # set the value of the weak learner to the parameter value, by default, it is set to Any
        self.n_elements = n_elements
        self.f = []  # these parameters were not stated in the constructor they are either declared in other functions related to the class and is most likely going to be optional parameters
        self.model_weights = [] 
        self.f1s = []
        self.mean_loss = []
        self.record_training_metrics = record_training_metrics   # set the value of the record_training_metrics to the record_training_metrics which is a boolean value that is set to false by default
    #destructor
    #the line defines a special method called __del__, which is a destructor method in python classes. A destructor is called when an object is about to be destroyed or garbage collected
    def __del__(self) -> None:  # the destructor also returns nothing, meaning no return statement is neccessary --> this is a private function
        del self.weak_learner
        del self.n_elements
        del self.f
        del self.model_weights
        del self.f1s
        del self.mean_loss 
        del self.record_training
        
# now let's define the two private loss functions for this class
    def __compute_loss(self, y_pred : np.array, y_train : np.array) -> np.array:  # this function is expected to return a numpy array. Additionally, this line defines a function named __compute_loss, the function takes two parameters, y_pred that represents the predicted values and y_train that represents the actual (true) values and the function is expected to return a numpy array
    # compute the loss function
        loss = (y_pred != y_train).astype(int)  # This line computes the loss by comparing each element of the 'y_pred' with the corresponding element of y_train. It creates a boolean array where True indicates a mismtach between the predicted and actual/true values and astype(int) converts True : 1 and False : 0 and it gets stored in the variable loss
    # return computed loss
        return(loss)  # returns the numpy array that has just been created

#private function to compute model weights
    def __compute_alpha(self, w : np.array, loss : np.array) -> float:   # this line defines a function named __compute_alpha, which takes in two parameters. w : represents a numpy array of weighs and loss which represents an array of loss values that was calculated using the above function
    
    #compute the error rate
        err = np.sum(np.multiply(w, loss)) / np.sum(w)  # error rate computation, this line calculates the error using the weighted sum of the losses. The np.multiply(w, loss) performs element-wise multiplication between the array 'w' and the loss array (think in terms of cross product). np.sum(np.multuply(w, loss)) calculates the sum of the element-wise product. '/ np.sum(w)' divides the result by the sum of the weights producing the error rate.
    #compute the adaptive weight. (this is eseentially the methemtical implementation of the error formula using numpy operations)
        alpha = np.log((1-err)/err)    # this line calculates the an adaptive weight (alpha for short) based on the computed error rate. np.log computes the natural logarithm of the expression ((1-err)/err) with err being what we calculated using the previous line's formula 
    #store alpha
        self.model_weights.append(alpha)  # storage of alpha (recall that we initialized model_weights as an empty array in the constructor function). The line appends the calculated 'alpha' value to a list named model_weights. Model_weights is a list used to store the adaptive weights during the training process.
    #return the computer alpha
        return (alpha)   # this line returns the calculated alpha value as the output of the function. The computed alpha is likely used in subsequent steps of an ensemble learning algorithm, such as AdaBoost, where weights are adjusted to emphasize or deemphasize the influence of weak learners based on their performance

#private function to compute training metrics for n-trained weak learners --> a function used to record performance during training. The metrics recorded include the F1 score, along with the loss l
    def __compute_training_metrics(self, X_train : np.array, y_train : np.array) -> Tuple[float, float]:    # this line defines a function named '__compute_training_metrics'. The function takes two parameters : 'X_train', representing the training features, and y_train, representing the true labels and lastly, the function is expected to return a tuple of two floating point numbers (remember that the difference between a tuple and a list is that a list is immutable but a tuple isn't)
    #initialize output
        y_pred = np.zeros((X_train.shape[0]))   # initialization of predictions with 0s that matches the size of X_train data (in simpler terms: this line initializes an array y_pred filled with zeros, having the same number of rows as the training features (X_train.shape[0])). y_pred will be used to accumulate predictions from the ensemble of models
    #traverse ensemble to generate predictions
        for model, mw in zip(self.f, self.model_weights):   # this block uses a loop to traverse the enseble of models (self.f) along with their corresponding weights ('self.model_weights'). For each model, it predicts on the training features ('model.predict(X_train)') and scales the predictions by its weight ('mw'). The scaled predictions are accumulated in the 'y_pred' array
            y_pred += mw * model.predict(X_train)
    
        #perform sign operation
        y_pred = np.round(y_pred).astype(int)   # prediction refinement --> this block of code refines the prediction by rounding them to the nearest integer (np.round(y_pred)) and then applying a sign of operations. Predictions less than or equal to 0 are set to -1, and predictions greater than 0 are set to 1
        y_pred = np.where(y_pred <= 0, -1, 1)
    
        #compute metrics
        f1 = f1_score(y_train, y_pred)   # this block of code calculates two performance metrics on the training set: 1. 'f1' (also known as f1 score), a measure of a model's accuracy, balancing precision and recall. 
        loss = np.mean(self.__compute_loss(y_pred, y_train))  # the average loss calculated by the '__compute_loss' method, which likely represents the misclassification rate
        #return computed metrics
        return(f1, loss)   # this line returns a tuple containing the calculated metrics ('f1' && 'loss') as the output of the function

    
   # In simpler terms, the function takes the training features and labels uses an ensemble of models to make predictions, refines those predictions, and then computes and returns two training metrics: the F1 score and a loss metric. These metrics help evalulate the performance of the ensemble on the training set.
    
   # We can now proceed with our training procedure
    
    
    #public function to train the ensemble
    def fit(self, X_train : np.array, y_train : np.array) -> None:   # this is a public function used to execute the training procedure outlined in the previous section. based upon the training data, these data consists of model input predictors (X_train) and labels (y_train). Additionally, this line defines a public method named 'fit' and this method is used to execute the training procedure based on the training data (X_train, y_train) that are both of numpy array (meaning our dataset first needs to be converted to numpy array datatype before we can use them as parameter values in our models)
    # check that y_train consists of {-1,1}
        if (np.unique(y_train).shape[0] != 2) or (np.min(y_train) != -1) or (np.max(y_train) != 1):   # this block of conditional statement is used to check whether the tables (y_train) are correctly formatted for a binary classification task (we cannot train this model with more than two labels). It also ensures that there are exactly two unique labels, and the minimum and maximum values are -1 and 1, respectively. If the conditions are not met, an exception is raised indicating a format error in the training labels.
            raise Exception('Training labels are not formatted to {-1,+1}')
    # initialize sample weights, residuals & models array
        w = np.ones((y_train.shape[0])) / y_train.shape[0]   # this block initiailzies the sample weights w to be uniform initially
        self.residuals = []     # self.residuals is initialized as an empty list. It seems like this list will be used to store residuals during the training process
        self.f = []   # is initialized as an empty list. It will be used to store the weak learners (models) during the training process.
    # loop through the specified number of iterations in the ensemble
        for _ in range(self.n_elements):   # iteration through ensembles --> this loop iterates through the specified number of elements (self.n_elements) in the ensemble
        # make a copy of the weak learner
            model = clone(self.weak_learner)   # this line creates a copy of the weak learner (self.weak_learner) using the 'clone' function . This is done to avoid modifying the original weak learner during training (the clone function has been imported from sklearn.base module)
        # fit the current weak learner on the dataset
            model.fit(X_train, y_train, sample_weight=w)   # this line trains the current weak learner ('model') on the training data ('X_train', 'y_train') with the specified sample ('w')
        #obtain predictions from the current weak learner
            y_pred = model.predict(X_train)   # this line obtains predictions from the current weak learner ('model') on the training data
        #compute the loss 
            loss = self.__compute_loss(y_pred, y_train)  # this line computes the loss using the predictions ('y_pred') and true labels (y_train) with the __compute_loss method (defined above as a private function)
        #compute the adaptive weight
            alpha = self.__compute_alpha(w, loss)  # this line calculates an adaptive weight ('alpha') based on the computed loss, using the __compute_alpha method (also a private function that was defined above) 
        #update sample weights
            w *= np.exp(alpha*loss)   # this line updates the sample weights ('w') based on the calculated adaptive weight ('alpha') and the loss
        #append resulting model
            self.f.append(model)   # this line appends the trained weak learner ('model') into the list of models ('self.f') 
        # append current training metrics
            if self.record_training:      # this block checks whether to record training metrics ('self.record_training') is a boolean_flag (recall that we initialised it as type boolean and the default value being False). If recording is enabled, it calculates f1 score and mean_loss and appends them to their respective lists ('self.f1s' and 'self.mean_loss') 
                f1, mean_loss = self.__compute_training_metrics(X_train, y_train)  # refer back to the above function __compute_training_metrics as it returns two values, to refer to private functions within classes, you need to use self.(name of function) being called upon
                self.f1s.append(f1)
                self.mean_loss.append(mean_loss)
    
    # finally, we can cover the remaining functions for its implementation
    # public function to return training f1 score
    def get_f1s(self) -> List:   # this is a getter function --> detailed explanation: this function is a getter method that retrieves and returns the list of F1 scores ('self.f1s'). F1 scores are likely recorded during the training process and can be useful for evaluating model's performance
        return (self.f1s)

    # public function to return mean training loss
    def get_loss(self) -> List:  # mean_loss getter function. This function is another getter method that retrieves and returns the list of mean training losses ('self.mean_loss'). mean_losses represents the average loss over the training iterations and can be used to assess the training performance
        return (self.mean_loss)

    # public function to return model parameters
    def get_params(self, deep : bool, bool = False) -> Dict:   # This function is a getter method that retrieves and returns a dictionary of model parameters. The returned dictionary includes information about the weak larner. The returned dictionary includes information about the weak learner (self.weak_learner) , the number of elements in the ensemble (n_elements) , and whether the training metrics is recorded or not (self.record_training_metrics)
        return {'weak_learner': self.weak_learner,
                'n_elements':self.n_elements,
                'record_training_metrics':self.record_training_metrics}
                
    # public function to generate predictions
    def predict(self, X_test : np.array) -> np.array:   # prediction generator --> the function generates predictions on new data ('X_test') using the trained ensemble of weak learners -->  it initializes an array ('y_pred') to store the predictions. It traverses through the ensemble, making predictions with each weak learner and scaling them by their corresponding weights ('mw') --> the predictions are then refined by rounding and a sign operation and lastly the final predictions are returned as an numpy array
        # initialize output
        y_pred = np.zeros((X_test.shape[0]))
        
        #traverse ensemble to generate predictions
        for model,mw in zip(self.f,self.model_weights):
            y_pred += mw * model.predict(X_test)
            
        #perform sign operation
        y_pred = np.round(y_pred).astype(int)
        y_pred = np.where(y_pred <= 0, -1, 1)
        
        # return predictions
        return(y_pred)
    
        
# note: to compile and run the code, use python3 AdaBoostClassifier.py