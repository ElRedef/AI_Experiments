import neuron


w = [[20,2,2],[-20,-2,2],[40,4,4]]
i = [2,20,20]

n = neuron.Neuron(w[0])
y = n.calculate(i)
print(y)


network = []

for x in range (3):
    network.append(neuron.Neuron(w[x]))

y = network[2].calculate(i)

print (y)