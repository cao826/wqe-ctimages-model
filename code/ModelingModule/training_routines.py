"""
Module Level Docstring
"""
import numpy as np
from torch import nn
import torch

def train_on_batch(batch, model, loss_fn, optimizer):
    """Trains the model on a batch of the data"""
    model.train()
    inputs, clinical_info_batch, labels = batch
    inputs = inputs.cuda()
    clinical_info_batch = clinical_info_batch.cuda()
    labels = labels.cuda()

    optimizer.zero_grad()
    outputs = model(inputs, clinical_info_batch)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    return loss.item()

def train_one_epoch(model, training_loader, loss_fn, optimizer): #COMPLETE
    """Trains the model for one epoch"""
    losses = []

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for _, batch in enumerate(training_loader):

        # Gather data and report
        losses.append( train_on_batch(
            batch = batch,
            model = model,
            loss_fn = loss_fn,
            optimizer = optimizer
            )
        )

    return losses

def process_losses(losses, epoch_num=None, mode='training'):
    """
    Return the mean loss over the entire batch
    """
    mean_loss = np.array(losses).mean()
    if epoch_num:
        print(f'Mean {mode} loss on epoch {epoch_num}: {mean_loss}')
    else:
        print(f'Mean {mode} training loss: {mean_loss}')
    return mean_loss

def eval_one_batch(batch, model, loss_fn):
    """Evaluates the model on one batch of the validation dataloader"""
    #print(type(batch))
    #print(batch.shape)
    #print(batch)
    model.eval()
    inputs, clinical_info, labels = batch
    inputs = inputs.cuda()
    clinical_info = clinical_info.cuda()
    labels = labels.cuda()

    outputs = model(inputs, clinical_info)
    loss = loss_fn(outputs, labels)

    return loss.item()

def eval_on_dataset(model, eval_loader, loss_fn):
    """Evaluates the model on the validation data"""
    losses = []
    for _, batch in enumerate(eval_loader):
        losses.append(
            eval_one_batch(
                batch = batch,
                model = model,
                loss_fn = loss_fn
            )
        )
    return losses

def logits_to_preds_cross(logits_tensor):
    """Converts the calss scores output by model to class probabilities"""
    probs = nn.Softmax(dim=1)(
        logits_tensor.detach()
        )
    preds = torch.argmax(
        probs,
        dim=1
    )
    return preds
