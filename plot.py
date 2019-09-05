import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_fig(x, y, curve_name=None, xlabel='epoch', ylabel='reward or loss', filename='training_jpg', title='training_progress'):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if curve_name == None:
        plt.plot(x, y)
    else:
        plt.plot(x, y, label=curve_name)
    #plt.legend(bbox_to_anchor=(0.2, 0.8), loc=4, borderaxespad=0)
    plt.legend()

def read_log(args, log_filename, ret_content='reward'):
    with open(log_filename, 'r') as f:
        data = f.read().split('\n')
    content = list()
    for index, line in enumerate(list(filter(None, data))):
        if index > args.max_epoch:
            break
        content.append(list(filter(None, line.split(', '))))
    content = np.array(content, dtype='f')
    if ret_content == 'reward':
        return content[:,0].tolist(), content[:,2].tolist()
    else:#'loss'
        return content[:,0].tolist(), content[:,1].tolist()

def main(args):
    filename = ''
    for index, log_filename in enumerate(args.plot_name):
        if index != 0:
            filename += '__'
        filename += log_filename.split('.')[0]
        if args.plot_reward:
            x, y = read_log(args, log_filename, 'reward')
            plot_fig(x, y, curve_name=log_filename.split('.')[0])
        if args.plot_loss:
            x, y = read_log(log_filename, 'loss')
            plot_fig(x, y, curve_name=log_filename.split('.')[0])
        print(len(x), len(y))
    filename += '.jpg'
    plt.savefig(filename)
    plt.clf()



if __name__ == '__main__':
    def str2bool(v):
        if v == 'True':
            return True
        return False
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_name', nargs='*', type=str)
    parser.add_argument('--plot_reward', default=True, type=str2bool)
    parser.add_argument('--plot_loss', default=False, type=str2bool)
    parser.add_argument('--max_epoch', default=500, type=int)
    args = parser.parse_args()
    print(len(args.plot_name), ":", args.plot_name)
    main(args)
