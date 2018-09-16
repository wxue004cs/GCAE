import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np


def train(train_iter, dev_iter, mixed_test_iter, model, args, text_field, aspect_field, sm_field, predict_iter):
    time_stamps = []

    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.l2, lr_decay=args.lr_decay)

    steps = 0
    model.train()
    start_time = time.time()
    dev_acc, mixed_acc = 0, 0
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, aspect, target = batch.text, batch.aspect, batch.sentiment

            feature.data.t_()
            if len(feature) < 2:
                continue
            if not args.aspect_phrase:
                aspect.data.unsqueeze_(0)
            aspect.data.t_()
            target.data.sub_(1)  # batch first, index align

            if args.cuda:
                feature, aspect, target = feature.cuda(), aspect.cuda(), target.cuda()

            optimizer.zero_grad()
            logit, _, _ = model(feature, aspect)

            loss = F.cross_entropy(logit, target)
            loss.backward()

            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                if args.verbose == 1:
                    sys.stdout.write(
                        '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                                 loss.data[0],
                                                                                 accuracy,
                                                                                 corrects,
                                                                                 batch.batch_size))


            if steps % args.save_interval == 0:
                if not os.path.isdir(args.save_dir): os.makedirs(args.save_dir)
                save_prefix = os.path.join(args.save_dir, 'snapshot')
                save_path = '{}_steps{}.pt'.format(save_prefix, steps)
                torch.save(model, save_path)

        if epoch == args.epochs:
            dev_acc, _, _ = eval(dev_iter, model, args)
            if mixed_test_iter:
                mixed_acc, _, _ = eval(mixed_test_iter, model, args)
            else:
                mixed_acc = 0.0

            if args.verbose == 1:
                delta_time = time.time() - start_time
                print('\n{:.4f} - {:.4f} - {:.4f}'.format(dev_acc, mixed_acc, delta_time))
                time_stamps.append((dev_acc, delta_time))
                print()
    return (dev_acc, mixed_acc), time_stamps


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    loss = None
    for batch in data_iter:
        feature, aspect, target = batch.text, batch.aspect, batch.sentiment
        feature.data.t_()
        if not args.aspect_phrase:
            aspect.data.unsqueeze_(0)
        aspect.data.t_()
        target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, aspect, target = feature.cuda(), aspect.cuda(), target.cuda()

        logit, pooling_input, relu_weights = model(feature, aspect)
        loss = F.cross_entropy(logit, target, size_average=False)
        avg_loss += loss.data[0]
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss = loss.data[0]/size
    accuracy = 100.0 * corrects/size
    model.train()
    if args.verbose > 1:
        print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(
           avg_loss, accuracy, corrects, size))
    return accuracy, pooling_input, relu_weights
