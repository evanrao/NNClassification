# coding=utf-8
class Neural:
    def __init__(self,si,xi,ui,oi):
        print('I\'m Neural Network,init now!')
        self.Si = si #The number of samples
        self.Xi = xi # Enter the number of variables
        self.Ui = ui # Enter the number of membership functions for each input variable
        # Hi <= Ui ** Xi ; Hidden layer refers to the number of rules here, Rules, the number of hidden nodes
        self.Hi = self.Ui ** self.Xi
        self.Oi = oi # The number of output nodes

