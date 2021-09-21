from mnist import MNIST
import random
import numpy as np

def sigma(z):
        #helper sigmoid function which squishes a number between 0 and 1
    return 1.00/(1.00+np.exp(-z))

def sigma_derivative(z):
    return (np.exp(-z))/(pow(1.00+np.exp(-z), 2))

def reLU(z):
    if z>0:
        return z
    else:
        return 0.01 * z

def reLU_matrix(M):
    for i in range(len(M)):
        for j in range(len(M[0])):
            M[i][j] = reLU(M[i][j])
    return M
def reLU_derivative(z):
    if z> 0:
        return 1
    else:
        return 0.01

def reLU_derivative_matrix(M):
    for i in range(len(M)):
        for j in range(len(M[0])):
            M[i][j] = reLU_derivative(M[i][j])
    return M

def label_to_des_output(label):
    des_output = [0] * 10
    des_output[label] = 1
    return des_output

class neural_network:
    #Will consist of 784 neurons connected to two 'hidden' layers of 16 neurons each
    def __init__(self):


        #Initializing all weights and biases

        self.inpto1_weight_matrix = np.random.randn(784, 16)
        self.first_bias_matrix = np.random.randn(1,16)

        self.firsttosecond_weight_matrix = np.random.randn(16, 16)
        self.second_bias_matrix = np.random.randn(1,16)

        self.secondtoout_weight_matrix = np.random.randn(16,10)
        self.third_bias_matrix = np.random.randn(1,10)

        self.weights = [self.inpto1_weight_matrix, self.firsttosecond_weight_matrix, self.secondtoout_weight_matrix]

        self.biases = [self.first_bias_matrix, self.second_bias_matrix, self.third_bias_matrix]

        self.acts_layer_sizes = [(1, 784), (1, 16), (1, 16), (1, 10)]
        self.weights_layer_sizes = [(784, 16), (16, 16), (16, 10)]
        self.biases_layer_sizes = self.acts_layer_sizes


    def get_layer_activations(self, input_activations, layer):

        if layer == 0:
            return input_activations

        last_activations = self.get_layer_activations(input_activations, layer - 1)

        activations = sigma(np.matmul(last_activations, self.weights[layer - 1] ) + self.biases[layer-1])

        return activations

    def get_z_layer(self, input_activations, layer):
        #Assume layer != 0
        last_activations = self.get_layer_activations(input_activations, layer - 1)

        z_layer = np.matmul(last_activations, self.weights[layer - 1] ) + self.biases[layer-1]

        return z_layer

    def get_z_list(self, activations, input_activations):
            z_list = [np.zeros((1, 784))]
            for layer in range(1, len(activations), 1):
                z_list.append(self.get_z_layer(input_activations, layer))
            return z_list



    def get_out_acts_grad(self, activations, des_output):
        grads = np.zeros((1,10))

        for out_act in range(10):
            grads[0][out_act] = 2*(activations[3][0][out_act] - des_output[out_act])

        return grads

    def grad_weight_layer(self, z_list, activations, activations_gradient, layer):

        size = self.weights_layer_sizes[layer]
        grads = np.zeros(size)

        for weight_group in range(size[1]):

            z = z_list[layer + 1][0][weight_group]

            for weight in range(size[0]):
                grads[weight][weight_group] = activations[layer][0][weight] *sigma_derivative(z) * activations_gradient[layer + 1][0][weight_group]

        return grads

    def grad_bias_layer(self, z_list, activations, activations_gradient, layer):
        size = self.biases_layer_sizes[layer]
        grads = np.zeros(size)

        for bias in range(size[1]):
            z = z_list[layer][0][bias]
            grads[0][bias] = sigma_derivative(z) * activations_gradient[layer][0][bias]

        return grads

    def grad_acts_layer(self, z_list, activations_gradient, layer):

        size = self.acts_layer_sizes[layer]
        weights_size = self.weights_layer_sizes[layer]

        grads = np.zeros(size)
        for activation in range(size[1]):
            for weight in range(weights_size[1]):
                z = z_list[layer + 1][0][weight]
                grads[0][activation] = grads[0][activation] + (self.weights[layer][activation][weight] * sigma_derivative(z) * activations_gradient[layer+1][0][weight])

        return grads




    def gradient_for_one(self, input_activations, label):

        #Convert the label into an array of 1's and 0's to make the desired output in the cost fcn
        des_output =  label_to_des_output(label)
        input_activations = input_activations * 1/255
        #Calculate and store all the activations
        l1_acts = self.get_layer_activations(input_activations, 1)
        l2_acts = self.get_layer_activations(input_activations, 2)
        lout_acts = self.get_layer_activations(input_activations, 3)
        activations = [input_activations, l1_acts, l2_acts, lout_acts]

        #Get the list of the z(w,b) function results for later use
        z_list = self.get_z_list(activations, input_activations)

        #initialize lists of all the gradients
        activations_gradient = [np.zeros((1, 784)), np.zeros((1, 16)), np.zeros((1, 16)), np.zeros((1, 10))]

        weights_gradient = [np.zeros((784,16)), np.zeros((16, 16)), np.zeros((16, 10))]

        biases_gradient = [np.zeros((1,16)), np.zeros((1,16)), np.zeros((1,10))]

        #Calculate the easy case of the gradient of the output activations

        activations_gradient[3] = self.get_out_acts_grad(activations, des_output)


        #Now use these to find the gradient of the weights and biases between layer 2 and 3

        weights_gradient[2] = self.grad_weight_layer(z_list, activations, activations_gradient, 2)
        biases_gradient[2] = self.grad_bias_layer(z_list, activations, activations_gradient, 3)



        #Calculate the gradient of the second layer neurons activations and store it correctly

        activations_gradient[2] = self.grad_acts_layer(z_list, activations_gradient, 2)

        #Now use these to find the gradient of the weights and biases between layer 1 and 2

        weights_gradient[1] = self.grad_weight_layer(z_list, activations, activations_gradient, 1)
        biases_gradient[1] = self.grad_bias_layer(z_list, activations, activations_gradient, 2)

        #Calculate the gradient of the first layer neurons
        activations_gradient[1] = self.grad_acts_layer(z_list, activations_gradient, 1)

        #Now use these to find the gradient of the weight and biases between layer input and 1

        weights_gradient[0] = self.grad_weight_layer(z_list, activations, activations_gradient, 0)
        biases_gradient[0] = self.grad_bias_layer(z_list, activations, activations_gradient, 1)

        gradient_for_one = {"weights": weights_gradient, "biases": biases_gradient}

        return gradient_for_one

    def train(self, images, labels):

        for index in range(len(images)):
            #print(index)
            image = images[index]

            label = labels[index]

            gradient_for_one = self.gradient_for_one(np.array([image]), label)

            self.update_w_and_b(gradient_for_one)

            print("Training on data:",index)

        return

    def update_w_and_b(self, avg_grad):

        self.inpto1_weight_matrix -= avg_grad["weights"][0]
        self.firsttosecond_weight_matrix -= avg_grad["weights"][1]
        self.secondtoout_weight_matrix -= avg_grad["weights"][2]

        self.first_bias_matrix -= avg_grad["biases"][0]
        self.second_bias_matrix -= avg_grad["biases"][1]
        self.third_bias_matrix -= avg_grad["biases"][2]

        self.weights = [self.inpto1_weight_matrix, self.firsttosecond_weight_matrix, self.secondtoout_weight_matrix]

        self.biases = [self.first_bias_matrix, self.second_bias_matrix, self.third_bias_matrix]

        return

    def cost_fcn(self, input_activations, label):

        des_output = label_to_des_output(label)

        res = 0

        for i in range(len(des_output)):
            res = res + pow((des_output[i] - input_activations[0][i]), 2)

        return res



    def give_answer(self, image):
        input_activations = np.array([image])
        input_activations = input_activations * 1/255
        out_acts = self.get_layer_activations(input_activations, 3)
        min_cost = None
        answer = None
        for index in range(10):
            cost = self.cost_fcn(out_acts, index)

            if min_cost == None:

                min_cost = cost
                answer = index
                continue

            if cost < min_cost:
                min_cost = cost
                answer = index

        return answer

    def test_one(self, test_image, label):

        answer = self.give_answer(test_image)

        if answer == label:
            return 1

        return 0

    def run_tests(self, test_images, labels):
        total = len(test_images)
        correct = 0

        for i in range(len(test_images)):
            test_image = test_images[i]
            label = labels[i]
            if self.test_one(test_image, label) == 1:
                correct += 1

        return correct/total



if __name__ == "__main__":
    mndata = MNIST('TrainingTestData')

    images, labels = mndata.load_training()
# or
    test_images, test_labels = mndata.load_testing()
    index = 0 #random.randrange(0, len(images))  # choose an index ;-)
    print(mndata.display(images[index]))
    print("Label is: " ,labels[index])
    nn = neural_network()
    #print(nn.weights[1])
    nn.train(images, labels)
    print(nn.run_tests(test_images, test_labels))
    #print(nn.weights[1])
    #print("Answer is: ", nn.give_answer(images[index]))
    #print("Result is: ", nn.test_one(images[index], labels[index]))

    #print(nn.weights)
    #print(nn.get_layer_activations(np.array(images[index]), 2))