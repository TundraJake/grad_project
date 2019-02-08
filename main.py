'''

Jacob McKenna
UAF Graduate Project
main.py - Starting point of the entire project.

'''
from neural_network import *
from online_data_fetcher import * 

X = np.array([1,2,3])

def main():
    

    X = np.array([[1,1,1,1,1,1,0,0,0,0,1,1],
                    [1,1,1,1,1,1,0,0,0,0,1,1],
                    [1,1,1,1,1,1,0,0,0,0,1,1],
                    [1,1,1,0,1,1,0,0,0,0,1,1],
                    [1,0,1,1,1,1,0,0,0,0,1,1],
                    [1,1,0,1,0,1,0,0,0,0,1,1],
                    [1,1,1,1,0,1,0,0,0,0,1,1]])

    Y = np.array([1,0,0,1,1,0,1,])

    example_nn = Neural_Network(X.shape[1])
    example_nn.load_data(X, Y)
    example_nn.train()

if __name__ == "__main__":
    main()



