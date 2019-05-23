import torch
import numpy as np
from collections import OrderedDict
from torch import optim
from base import Base

class Classifier(Base):
    def __init__(self,
                 classifier,
                 opt=optim.Adam,
                 opt_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
                 handlers=[],
                 scheduler_fn=None,
                 scheduler_args={}):
        use_cuda = True if torch.cuda.is_available() else False
        self.classifier = classifier
        optim_cls = opt(filter(lambda p: p.requires_grad,
                               self.classifier.parameters()), **opt_args)
        self.optim = {
            'cls': optim_cls,
        }
        self.scheduler = {}
        if scheduler_fn is not None:
            for key in self.optim:
                self.scheduler[key] = scheduler_fn(
                    self.optim[key], **scheduler_args)
        self.handlers = handlers
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.classifier.cuda()
        self.last_epoch = 0

    def _get_stats(self, dict_, mode):
        stats = OrderedDict({})
        for key in dict_.keys():
            stats[key] = np.mean(dict_[key])
        return stats
    
    def _train(self):
        self.classifier.train()

    def _eval(self):
        self.classifier.eval()

    def bce(self, prediction, target):
        if not hasattr(target, '__len__'):
            target = torch.ones_like(prediction)*target
            if prediction.is_cuda:
                target = target.cuda()
        loss = torch.nn.BCELoss()
        if prediction.is_cuda:
            loss = loss.cuda()
        target = target.view(-1, 1)
        return loss(prediction, target)
        
    def train_on_instance(self,
                          x_batch,
                          y_batch,
                          **kwargs):
        self._train()
        for key in self.optim:
            self.optim[key].zero_grad()

        preds = self.classifier(x_batch)
        with torch.no_grad():
            tot_acc = ((preds >= 0.5).float() == y_batch).float().mean()
        
        loss_fn = self.bce(preds, y_batch)
        loss_fn.backward()
        self.optim['cls'].step()
                
        losses = {
            'loss': loss_fn.item(),
            'tot_acc': tot_acc.item()
        }
        outputs = {}
        return losses, outputs

    def eval_on_instance(self,
                         x_batch,
                         y_batch,
                         **kwargs):
        self._eval()
        with torch.no_grad():
            preds = self.classifier(x_batch)
            tot_acc = ((preds >= 0.5).float() == y_batch).float().mean()
            loss_fn = self.bce(preds, y_batch)
            losses = {
                'loss': loss_fn.item(),
                'tot_acc': tot_acc.item()
            }
            
        return losses, {}

    def prepare_batch(self, batch):
        if len(batch) != 3:
            raise Exception("Expected batch to only contain three elements: " +
                            "X_batch, y_batch_1, and y_batch_2")
        X_batch = batch[0].float()
        y_batch_1 = batch[1]
        y_batch_2 = batch[2]
        y_batch = torch.cat((y_batch_1, y_batch_2), dim=1).float()
        
        if self.use_cuda:
            X_batch = X_batch.cuda()
            y_batch = y_batch.cuda()
        return [X_batch, y_batch]

    def save(self, filename, epoch):
        dd = {}
        # Save the models.
        dd['cls'] = self.classifier.state_dict()
        # Save the models' optim state.
        for key in self.optim:
            dd['optim_%s' % key] = self.optim[key].state_dict()
        dd['epoch'] = epoch
        torch.save(dd, filename)

    def load(self, filename):
        if not self.use_cuda:
            map_location = lambda storage, loc: storage
        else:
            map_location = None
        dd = torch.load(filename,
                        map_location=map_location)
        # Load the models.
        self.classifier.load_state_dict(dd['cls'])
        # Load the models' optim state.
        for key in self.optim:
            self.optim[key].load_state_dict(dd['optim_%s' % key])
        self.last_epoch = dd['epoch']
