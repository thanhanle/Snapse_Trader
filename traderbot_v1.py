"""
Stock market predicting using a feed-forward neural network.
"""

from __future__ import print_function
import os
import neat
import visualize
import train_window
import rand_train_window
import multi_rand_trainer
import numpy
import pandas as pd


prices = [pd.read_csv('EWY.csv')[:-500],pd.read_csv('QQQ.csv')[:-500],pd.read_csv('USO.csv')[:-500]]
test_prices = [pd.read_csv('EWY.csv')[-500:],pd.read_csv('QQQ.csv')[-500:],pd.read_csv('USO.csv')[-500:]]

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def eval_genome(genome, config):
    #net = neat.nn.FeedForwardNetwork.create(genome, config)
    net = neat.nn.RecurrentNetwork.create(genome, config)
    fittness = evaluation(net,prices,False)
    #fittness = randeval(net,prices)
    return fittness

def randeval(net,prices):
    st = multi_rand_trainer.sliding_trainer(prices,10,50)
    inputs,tests = [],[]
    startingcash = 10000
    cash = 10000
    portfolio = 0
    total = cash + portfolio
    for i in range(1):
        inputs,tests = st.random_train()
        output = net.activate(inputs)
        #print(output)
        if output[0] >= .5:
            portfolio = total  #*output[0]
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


def evaluation(net,prices,show):
    st = train_window.sliding_trainer(prices,10,100)
    inputs,tests = [],[]
    startingcash = 10000
    cash = 10000
    portfolio = 0
    total = cash + portfolio
    while inputs != "done":
        inputs,tests = st.slidestep()
        #print(inputs)
        if inputs != "done":
            output = net.activate(inputs)
            #if show == True:
            #    print(output)
        else:
            break
        if output[0] >= .5:
            portfolio = total #*output[0]
            cash = total - portfolio
        else:
            cash = total
            portfolio = 0
        delta = portfolio * ((tests[-1]-tests[0])/tests[0])
        #print(delta)
        total += delta
        if show == True:
            print(output, delta)
        cash = total
        portfolio = 0
    fittness = total
    return fittness 

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object [:-500]for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to n generations.
    winner = p.run(eval_genomes, 100)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    #winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    #for xi, xo in zip(xor_inputs, xor_outputs):
    #    output = winner_net.activate(xi)
    #    print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    print(evaluation(winner_net,test_prices,True))

    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    #p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)