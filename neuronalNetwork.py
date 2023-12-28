from matrice import Matrice
import math

def sigmoid(x : float):
    return 1 / (1 + math.exp(-x))

def dsigmoid(x : float):

    return sigmoid(x) * (1 - sigmoid(x))

class NeuronalNetwork :

    nbInputs : int
    nbHidden : int
    nbOutputs : int
    inputs : Matrice

    weights_1 : Matrice
    weights_2 : Matrice
    bias_1 : Matrice
    bias_2 : Matrice
    
    def __init__(self, nbInputs : int, nbHidden : int , nbOutputs) -> None:
        self.nbInputs = nbInputs
        self.nbHidden = nbHidden
        self.nbOutputs = nbOutputs

        self.weights_1 = Matrice.randomMatrice(nbHidden, nbInputs) 
        self.weights_2 = Matrice.randomMatrice(nbOutputs, nbHidden)
        self.bias_1 = Matrice.randomMatrice(nbHidden, 1 )
        self.bias_2 = Matrice.randomMatrice(nbOutputs, 1)

    def guess(self, inputs : list[float]):

        inputs_matrice = []
        for i in range(len(inputs)):
            inputs_matrice.append([inputs[i]])
        inputs_matrice = Matrice.matrice(inputs_matrice)
        #weights 1
        hidden_outputs = Matrice.multiply(self.weights_1, inputs_matrice)
        hidden_outputs = Matrice.add(hidden_outputs, self.bias_1)
        hidden_outputs_activated = Matrice.map(hidden_outputs, sigmoid)
        #weights 2
        output_outputs = Matrice.multiply(self.weights_2, hidden_outputs_activated)
        output_outputs = Matrice.add(output_outputs, self.bias_2)
        output_outputs_activated = Matrice.map(output_outputs, sigmoid)

        res = []
        for i in range(len(output_outputs_activated.matrice)):
            res.append(output_outputs_activated.matrice[i][0])
        return res

    
    def feedForward(self, inputs : list[float]):
        inputs_matrice = []
        for i in range(len(inputs)):
            inputs_matrice.append([inputs[i]])
        inputs_matrice = Matrice.matrice(inputs_matrice)
        #weights 1
        hidden_outputs = Matrice.multiply(self.weights_1, inputs_matrice)
        hidden_outputs = Matrice.add(hidden_outputs, self.bias_1)
        hidden_outputs_activated = Matrice.map(hidden_outputs, sigmoid)
        #weights 2
        output_outputs = Matrice.multiply(self.weights_2, hidden_outputs_activated)
        output_outputs = Matrice.add(output_outputs, self.bias_2)
        output_outputs_activated = Matrice.map(output_outputs, sigmoid)

        return inputs_matrice, hidden_outputs, hidden_outputs_activated, output_outputs, output_outputs_activated
        
    
    def train(self, inputs : list[float], answers: list[float]) :
        
        learnRate = 0.1

        inputs, hidden_outputs, hidden_outputs_activated, output_outputs, output_outputs_activated = self.feedForward(inputs)
        # output_outputs.print()
        anwers_matrice  = []
        for i in range(len(answers)):
            anwers_matrice.append([answers[i]])
        anwers_matrice = Matrice.matrice(anwers_matrice)
        
        cost_derivative = Matrice.multiplyNb(Matrice.substract(output_outputs_activated, anwers_matrice), 2)


        outputs_activated_derivative = Matrice.map(output_outputs, dsigmoid)
        # output_outputs_activated.print()
        # outputs_activated_derivative.print()
        # print("##########")
        cost_derivative_outputs = Matrice.multiplySimple(cost_derivative, outputs_activated_derivative)

        cost_derivative_outputs = Matrice.multiplyNb(cost_derivative_outputs, learnRate)
        # ajuster le bias
        self.bias_2 = Matrice.substract(self.bias_2, cost_derivative_outputs) 
        
        # ajuster les poids hidden-outputs
        hidden_outputs_activated_T = Matrice.transpose(hidden_outputs_activated)
        delta_weights_2 = Matrice.multiply(cost_derivative_outputs, hidden_outputs_activated_T)
  
        self.weights_2 = Matrice.substract(self.weights_2, delta_weights_2)

        weights_2_T = Matrice.transpose(self.weights_2)
        cost_derivative_hidden_outputs_activated = Matrice.multiply(weights_2_T, cost_derivative_outputs)
        cost_derivative_hidden_outputs = Matrice.multiplySimple(cost_derivative_hidden_outputs_activated, Matrice.map(hidden_outputs, dsigmoid))

        cost_derivative_hidden_outputs = Matrice.multiplyNb(cost_derivative_hidden_outputs, learnRate)
        #ajuster le bias 1
        self.bias_1 = Matrice.substract(self.bias_1, cost_derivative_hidden_outputs)

        #ajuster le poids inputs-hidden
        inputs_T = Matrice.transpose(inputs)
        delta_weights_1 = Matrice.multiply(cost_derivative_hidden_outputs, inputs_T)

        self.weights_1 = Matrice.substract(self.weights_1, delta_weights_1)
