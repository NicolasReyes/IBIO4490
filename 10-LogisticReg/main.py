
# read kaggle facial expression recognition challenge dataset (fer2013.csv)
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score
import pickle
import os
from urllib import request
def sigmoid(x):
    return 1/(1+np.exp(-x))

def get_data():
    # angry, disgust, fear, happy, sad, surprise, neutral
    with open("fer2013.csv") as f:
        content = f.readlines()

    lines = np.array(content)
    num_of_instances = lines.size
    print("number of instances: ",num_of_instances)
    print("instance length: ",len(lines[1].split(",")[1].split(" ")))

    x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []

    for i in range(1,num_of_instances):
        #import pdb; pdb.set_trace()
        emotion, img, usage = lines[i].split(",")
        
        pixels = np.array(img.split(" "), 'float32')
        emotion = 1 if int(emotion)==3 else 0 # Only for happiness
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)

    #------------------------------
    #data transformation for train and test sets
    #import pdb; pdb.set_trace()
    x_train = np.array(x_train, 'float64')
    y_train = np.array(y_train, 'float64')
    
    x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size = 0.33, random_state=42)
    x_test = np.array(x_test, 'float64')
    y_test = np.array(y_test, 'float64')

    x_train /= 255 #normalize inputs between [0, 1]
    x_test /= 255
    x_val /= 255
    #import pdb; pdb.set_trace()
    x_train = x_train.reshape(x_train.shape[0], 48, 48)
    x_val = x_val.reshape(x_val.shape[0], 48, 48)
    x_test = x_test.reshape(x_test.shape[0], 48, 48)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_val = y_val.reshape(y_val.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')

    # plt.hist(y_train, max(y_train)+1); plt.show()

    return x_train, y_train, x_val, y_val, x_test, y_test

class Model():
    def __init__(self):
        params = 48*48 # image reshape
        out = 1 # smile label
        self.lr = 0.00001 # Change if you want
        self.W = np.random.randn(params, out)
        self.b = np.random.randn(out)

    def forward(self, image):
        image = image.reshape(image.shape[0], -1)
        out = np.dot(image, self.W) + self.b
        return out

    def compute_loss(self, pred, gt):
        J = (-1/pred.shape[0]) * np.sum(np.multiply(gt, np.log(sigmoid(pred))) + np.multiply((1-gt), np.log(1 - sigmoid(pred))))
        return J

    def compute_gradient(self, image, pred, gt):
        image = image.reshape(image.shape[0], -1)
        W_grad = np.dot(image.T, pred-gt)/image.shape[0]
        self.W -= W_grad*self.lr

        b_grad = np.sum(pred-gt)/image.shape[0]
        self.b -= b_grad*self.lr
    
    #def store_model(self):
        

def train(model):
    x_train, y_train, x_val, y_val, _, _ = get_data()
    batch_size = 100 # Change if you want
    epochs = 10000 # Change if you want
    losses = np.zeros((epochs,))
    losses_test = np.zeros((epochs,))
    counter = 0
    prev_loss = 0;
    for i in range(epochs):
        loss = []
        for j in range(0,x_train.shape[0], batch_size):
            _x_train = x_train[j:j+batch_size]
            #import pdb; pdb.set_trace()
            _y_train = y_train[j:j+batch_size]
            out = model.forward(_x_train)
            loss.append(model.compute_loss(out, _y_train))
            model.compute_gradient(_x_train, out, _y_train)
        out = model.forward(x_val)                
        loss_test = model.compute_loss(out, y_val)
        #import pdb; pdb.set_trace()
        curr_loss = np.array(loss).mean()
        if curr_loss - prev_loss>0 and  j>0:
            counter = counter +1
        else:
            counter = 0
        losses[i] = curr_loss
        losses_test[i] = loss_test
        prev_loss = curr_loss
        if counter >5:
            print('5 iterations with no decrease in error, stopping program')
            print('Final Number of Epochs: ',str(j))
            break
        print('Epoch {:6d}: {:.5f} | test: {:.5f}'.format(i, np.array(loss).mean(), loss_test))
    #import pdb; pdb.set_trace()
    wait = input('Training done, please press ENTER to continue.')    
    plot(losses,losses_test)
    model_dict = {'W':model.W,'b':model.b}
    with open('10k_regression.pickle','wb') as f:
             pickle.dump(model_dict,f)

def plot(train_loss,test_loss): # Add arguments
    # CODE HERE
    # Save a pdf figure with train and test losses
    epochs = int(len(train_loss))
    x_epochs = range(0,epochs)
    #import pdb; pdb.set_trace()
    fig,ax = plt.subplots()
    
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    ax.plot(x_epochs,train_loss,'r',label = 'Train Loss')
    ax.plot(x_epochs,test_loss,'b',label = 'Validation Loss')
    legend = ax.legend(loc='upper right', shadow=False, fontsize='medium')
    fig.suptitle('Train and Validation loss for Logistic Regresion Model')
    plt.show()
    
    fig.savefig(str(epochs)+'_epochs_regression.pdf')

def demo():
    from skimage.transform import resize
    from skimage.io import imread
    model = Model()
    model_p = pickle.load(open('10k_regression.pickle','rb'))
    model.W = model_p['W']
    model.b = model_p['b']
    predictions = []

    wild_path = os.getcwd()+'/wild_images/'
    path_list = os.listdir(wild_path)
    annotations = [1,1,1,0,0,0]
    for i in range(len(path_list)):
        image = imread(wild_path+path_list[i],as_gray=True)
        image = resize(image,(48,48))
        image = image.reshape(2304)
        #import pdb; pdb.set_trace()
        out = np.dot(image, model.W) + model.b
        #import pdb; pdb.set_trace()
        predictions.append(np.around(out))
        
    #import pdb; pdb.set_trace()    
    score = accuracy_score(annotations,predictions)
    print('The prediction evaluation of the specified model in demo.')
    #print('F1-measure: '+str(F1))
    #print('Average Precision: '+str(AP))
    print('Accuracy Score:'+str(score))
    
def test():



    _, _, _, _, x_test, y_test = get_data()
    model = Model()
    model_p = pickle.load(open('10k_regression.pickle','rb'))
    model.W = model_p['W']
    model.b = model_p['b']
    #import pdb; pdb.set_trace()
    predictions = []
    
    for i in range(x_test.shape[0]):
        image = x_test[i]
        image = image.reshape(2304)
        #import pdb; pdb.set_trace()
        out = np.dot(image, model.W) + model.b
        #import pdb; pdb.set_trace()
        if out>1:
            predictions.append(np.floor(out))
        else:
            predictions.append(np.around(out))
    #loss_test = model.compute_loss(out, y_test)
    
    #F1 = f1_score(y_test,predictions)
    AP = average_precision_score(y_test,predictions)
    precision, recall, _ = precision_recall_curve(y_test,predictions)
    F1 = np.mean(2*((precision*recall)/(precision+recall)))
    print('The prediction evaluation of the specified model.')
    print('F1-measure: '+str(F1))
    print('Average Precision: '+str(AP))
    plt.figure()
    plt.plot(recall,precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Recall and Precision curve of Logistic Regresion Model')
    # YOU CODE HERE
    # Show some qualitative results and the total accuracy for the whole test set
    pass

if __name__ == '__main__':

    import argparse
    import zipfile
    if os.path.exists(os.getcwd() + '/' + 'fer2013.zip')==False:
        path_url = 'http://bcv001.uniandes.edu.co/fer2013.zip'
        path_url_open = request.urlopen(path_url)
        path_read = path_url_open.read()
        
        file_s =open(os.getcwd() + '/' + 'fer2013.zip','wb')
        file_s.write(path_read)
        file_s.close()
        
        zip_sg = zipfile.ZipFile(os.getcwd() + '/'+'fer2013.zip','r')
        zip_sg.extractall()
        zip_sg.close()
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test', action ='store_true')
    parser.add_argument('--demo', action ='store_true')
    
    opts = parser.parse_args()
    
    if opts.test:
        test()
    elif opts.demo:
        demo()
    else:
        model = Model()
        train(model)
	test()

