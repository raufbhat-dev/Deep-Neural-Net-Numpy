class Loss:
    def __init__(self, loss_func):
        self.loss_func = loss_func
        self.loss_derivative =  np.zeros(1)
        self.loss =  0.0
    
    def getLoss(self):
        if self.loss_func == 'meanSquared':
            def func(y_pred,y):
                error = y_pred - y
                self.loss = np.sum(np.diag(np.matmul(error,error.T))) / y.shape[-1]
                self.loss_derivative = error
        if self.loss_func == 'crossEntropy':
            def func(y_pred,y):
                epsilon = 1e-12 
                self.loss = -1*np.sum(np.multiply(y,np.log(y_pred+epsilon))+np.multiply((1-y),np.log(1- y_pred+epsilon)))
                self.loss_derivative = y_pred - y 
        return func
    
