class Neuron:

    def __init__(self,weights):
        #print ("------Intialisierung-------")
        self.weights = weights

    def calculate(self,input):
        assert (len(input) == len(self.weights)), "Weights und Input ungleich lang"

        sum = 0
        for i in range(len(input)):
            sum=sum+input[i]*self.weights[i]

        output=max(0,sum)
        return output
        
if __name__ == "__main__":
    w = [20,2,2]
    i = [2,20,20]

    n = Neuron(w)
    y = n.calculate(i)
    print(y)



