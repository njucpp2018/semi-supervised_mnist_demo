import os
import time
import torch
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from data import MNIST
from model import Model
from utils import Logger


if __name__ == '__main__':

    st = time.time()
    os.system('rm -rf ../save/')

    torch.manual_seed(0)
    use_gpu = True
    batch_size = 100
    nEpoch = 10

    dataset = MNIST('../data/mnist/')
    train_set = dataset.train()
    # supervised data
    supervised_indices = range(batch_size)
    supervised_train_set = Subset(train_set, supervised_indices)
    supervised_train_dataloader = DataLoader(
        supervised_train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=use_gpu,
        drop_last=False
    )
    for fixed_img, fixed_gt in supervised_train_dataloader:
        pass
    # unsupervised data
    unsupervised_indices = range(batch_size, len(train_set))
    unsupervised_train_set = Subset(train_set, unsupervised_indices)
    unsupervised_train_dataloader = DataLoader(
        unsupervised_train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=use_gpu,
        drop_last=False
    )
    test_dataloader = DataLoader(
        dataset.test(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=use_gpu,
        drop_last=False
    )
    supervised_model = Model(use_gpu=use_gpu)
    unsupervised_model = Model(use_gpu=use_gpu)
    unsupervised_model.net.load_state_dict(supervised_model.net.state_dict())
    logger = Logger('../save/log/')

    for epoch in range(nEpoch):
        for i, (img, gt) in enumerate(unsupervised_train_dataloader):
            # train supervised model
            loss = supervised_model.train(fixed_img, fixed_gt)
            logger.record('supervised_loss', loss)
            # train unsupervised model
            if unsupervised_model.test_accuracy(fixed_img, fixed_gt) < 1.:
                loss = unsupervised_model.train(fixed_img, fixed_gt)
            else:
                loss = unsupervised_model.train(img)
            logger.record('unsupervised_loss', loss)
            # save
            if i % 100 == 0:
                # calculate the accuracy of supervised model
                supervised_accuracy = 0.
                for n, (img, gt) in enumerate(test_dataloader):
                    supervised_accuracy += supervised_model.test_accuracy(img, gt)
                supervised_accuracy /= (n+1)
                logger.record('supervised_accuracy', supervised_accuracy)
                # calculate the accuracy of unsupervised model
                unsupervised_accuracy = 0.
                for n, (img, gt) in enumerate(test_dataloader):
                    unsupervised_accuracy += unsupervised_model.test_accuracy(img, gt)
                unsupervised_accuracy /= (n+1)
                logger.record('unsupervised_accuracy', unsupervised_accuracy)
                # save the accuracy curve
                logger.save_fig(
                    'supervised_accuracy',
                    'unsupervised_accuracy',
                    together=True,
                )
                print(
                    '%d_%d, supervised accuracy: %.4f, unsupervised accuracy: %.4f'
                    %(epoch, i, supervised_accuracy, unsupervised_accuracy)
                )

    et = time.time()
    print('train finished. total time: %.2f'%(et-st))
