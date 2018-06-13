"""
Stock market predicting using a feed-forward neural network.
"""

from __future__ import print_function
import os
import neat
import visualize
import train_window
import rand_train_window
import numpy
import pandas as pd

# 2-input XOR inputs and expected outputs.
#xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
#xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]
prices = pd.read_csv('price.csv')
prices2 = pd.read_csv('GLD.csv')

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)
    #for genome_id, genome in genomes:
    #    genome.fitness = 4.0
    #    net = neat.nn.FeedForwardNetwork.create(genome, config)
    #    for xi, xo in zip(xor_inputs, xor_outputs):
    #        output = net.activate(xi)
    #        genome.fitness -= (output[0] - xo[0]) ** 2

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    #fittness = evaluation(net,prices[:-700])
    fittness = randeval(net,prices[:-500])
    return fittness

def randeval(net,prices):
    st = rand_train_window.sliding_trainer(prices2["Close"],10,50)
    inputs,tests = [],[]
    startingcash = 10000
    cash = 10000
    portfolio = 0
    total = cash + portfolio
    for i in range(5):
        inputs,tests = st.random_train()
        output = net.activate(inputs)
        if output[0] >= .50:
            portfolio = total*output[0]
            cash = total - portfolio
        else:
            cash = total
            portfolio = 0
        delta = portfolio * ((tests[-1]-tests[0])/tests[0])
        total += delta
        cash = total
        portfolio = 0
    fittness = total
    return fittness

# For normal use!!! Don't edit!!
def evaluation(net,prices):
    st = train_window.sliding_trainer(prices["Close"],10,50)
    inputs,tests = [],[]
    startingcash = 10000
    cash = 10000
    portfolio = 0
    total = cash + portfolio
    while inputs != "done":
        inputs,tests = st.slidestep()
        if inputs != "done":
            output = net.activate(inputs)
        else:
            break
        if output[0] >= .50:
            portfolio = total*output[0]
            cash = total - portfolio
        else:
            cash = total
            portfolio = 0
        delta = portfolio * ((tests[-1]-tests[0])/tests[0])
        #print(delta)
        total += delta
        cash = total
        portfolio = 0
    fittness = total
    return fittness 

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to n generations.
    winner = p.run(eval_genomes, 200)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    #for xi, xo in zip(xor_inputs, xor_outputs):
    #    output = winner_net.activate(xi)
    #    print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    print(evaluation(winner_net,prices2[-500:]))

    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    #p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)