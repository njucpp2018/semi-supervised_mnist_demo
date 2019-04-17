import os
import json
import matplotlib.pyplot as plt


class Logger(object):

    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.data_dict = {}

    def record(self, *args):
        assert(len(args)%2 == 0)
        for i in range(len(args)//2):
            assert(type(args[i*2]) == str)
            assert(type(args[i*2+1]) == float or type(args[i*2+1]) == int)
            if args[i*2] in self.data_dict:
                self.data_dict[args[i*2]].append(args[i*2+1])
            else:
                self.data_dict[args[i*2]] = [args[i*2+1]]

    def save_fig(self, *keys, avg=1, together=False, set_name=None):
        if together:
            name = ''
        for key in keys:
            if not key in self.data_dict:
                continue
            n = len(self.data_dict[key]) // avg
            data = []
            for i in range(n):
                data.append(sum(self.data_dict[key][i*avg : (i+1)*avg]) / avg)
            plt.plot(range(n), data, label=key)
            plt.legend()
            if not together:
                plt.savefig(self.log_dir+key+'.jpg')
                plt.clf()
            else:
                name += ('+'+key)
        if together:
            if set_name is not None:
                name = set_name
            else:
                name = name[1:] + '.jpg'
            plt.savefig(self.log_dir+name)
            plt.clf()

    def save_json(self, *keys):
        for key in keys:
            if not key in self.data_dict:
                continue
            with open(self.log_dir+key+'.json', 'w') as f:
                json.dump(self.data_dict[key], f)

    def clear(self, *keys):
        if len(keys) == 0:
            self.data_dict = {}
        else:
            for key in keys:
                if key in self.data_dict:
                    self.data_dict.pop(key)
