import math
import numpy as np
class Nerual_Net:
    def __init__(self,num_node,Hidden_layer=1,lr=2,iteration=20):
        self.num_node=num_node
        self.Hidden_layer=Hidden_layer
        self.lr=lr
        self.iteration=iteration
    def Forward(self,X):
        if int(X[0])==1 or int(X[-1])==1:
            pass
        else:
            print('data don\'t have bias')
            X=list(X).append(1)
        self.HJ=self.weight_V.dot(X)#shape:[num_node,1]
        self.AJ=self.sigmoid(self.HJ)#shape:[num_node,1]
        # print(self.AJ.shape)
        self.target=self.sigmoid(self.weight_W.dot(self.AJ))
        # print(self.target)
        return self
    def BackWard(self,X,label):
        self.delat0=(label-self.target)*self.derivate(self.target)
        self.delta1 = self.AJ * (1 - self.AJ) * self.delat0 * self.weight_W
        self.delta1.shape = (1, len(self.delta1))
        self.weight_W+=self.lr*self.delat0*self.AJ
        X.shape=(1,len(X))
        self.weight_V+=self.lr*self.delta1.T.dot(X)
    def nerual_net_building(self,X):
        n,m=len(X),len(X[0])
        self.weight_V=np.array([[np.random.rand(1)[0] for _ in range(m)] for _ in range(self.num_node)])#Vji shape:[num_node,m]
        self.weight_W=np.array([np.random.rand(1)[0] for _ in range(self.num_node)])#Wkj shape:[Num_node,1]
        # return self
    def sigmoid(self,value):
        return 1/(1+np.exp(-value))
    def derivate(self,value):
        return value*(1-value)
    def training(self,X_train,Y_train):
        self.nerual_net_building(X_train)
        for _ in range(self.iteration):
            for X,label in zip(X_train,Y_train):
                self.Forward(X)
                self.BackWard(X,label)
            return self
    def predict(self,X_test,predict_prob=False):
        Y_pre=[]
        for line in X_test:
            Y_pre.append(self.Forward(line).target)
        if predict_prob:
            return np.array(Y_pre)
        else:
            return np.array([1 if x>0.5 else 0 for x in Y_pre])
    def evaulate(self,Y_pre,Y_true,metric='accuracy'):
        accuracy=0
        total=len(Y_pre)
        if metric=='accuracy':
            for x,y in zip(Y_pre,Y_true):
                # print(x,y)
                if x==y:
                    accuracy+=1
        return accuracy/total


NUM_FEATURES = 124 #features are 1 through 123 (123 only in test set), +1 for the bias
DATA_PATH = "e:/data set/a7a.test"
def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
    for t in tokens[1:]:
        parts = t.split(':')
        feature = int(parts[0])
        value = int(parts[1])
        x[feature-1] = value
    x[-1] = 1 #bias
    return y, x

#return labels and feature vectors for all datapoints in the given file
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_line(line) for line in f]
        (ys, xs) = ([v[0] for v in vals],[v[1] for v in vals])
        return np.asarray(ys), np.asarray(xs) #returns a tuple, first is an array of labels, second is an array of feature vectors



if __name__ == '__main__':
    test_ys, test_xs = parse_data(DATA_PATH)
    test_ys=[1 if x==1 else 0 for x in test_ys]
    F=Nerual_Net(num_node=5)
    F.training(test_xs,test_ys)
    print(F.evaulate(F.predict(test_xs),test_ys))
    # print(F.weight_W)
    # print('###############')
    # print(F.weight_V)
