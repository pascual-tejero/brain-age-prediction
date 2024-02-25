import torch
from torch import Tensor

from tqdm import tqdm

from utils.train_cnn_utils import AvgMeter, mean_absolute_error, seed_everything, TensorboardLogger
from utils.data_utils import get_image_dataloaders

import matplotlib.pyplot as plt
from argparse import Namespace
import os

from models.model import AgePredictionCNN

def config_model(model):
    # Set hyperparameters
    config = Namespace()
    config.img_size = 96
    config.batch_size = 16 ### None
    config.num_workers = 0

    config.log_dir = './logs'
    config.val_freq = 50
    config.log_freq = 10

    config.seed = 0
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    seed_everything(config.seed)

    config.lr = 0.0001
    config.betas = (0.9, 0.999)
    config.num_steps = 2500

    # Init optimizers
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas
    )

    # Init tensorboard
    writer = TensorboardLogger(config.log_dir, config)

    # Load data
    dataloaders = get_image_dataloaders(
        img_size=config.img_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    return config, writer, optimizer, dataloaders

def train_cnn(config, model, optimizer, train_loader, val_loader, writer):

    model.train()
    step = 0
    pbar = tqdm(total=config.val_freq, desc=f'Training') # Progress bar
    avg_loss = AvgMeter()

    while True:
        for x, y in train_loader:
            x, y = x.to(config.device), y.to(config.device)
            pbar.update(1) # Update progress bar

            # Forward pass
            optimizer.zero_grad()
            loss = model.train_step(x,y)
            loss.backward()
            optimizer.step()

            avg_loss.add(loss.item()) # Update loss

            step += 1 # Update step

            if step % config.log_freq == 0 and not step % config.val_freq == 0:
                train_loss = avg_loss.compute()
                writer.log({'train/loss': train_loss}, step=step)

            # Validate and log at validation frequency
            if step % config.val_freq == 0:
                # Reset avg_loss
                train_loss = avg_loss.compute()
                avg_loss = AvgMeter()

                # Get validation results
                val_results = validate(
                    model,
                    val_loader,
                    config,
                )

                # Print current performance
                print(f"Finished step {step} of {config.num_steps}. "
                      f"Train loss: {train_loss} - "
                      f"val loss: {val_results['val/loss']:.4f} - "
                      f"val MAE: {val_results['val/MAE']:.4f}")

                # Write to tensorboard
                writer.log(val_results, step=step)

                # Reset progress bar
                pbar = tqdm(total=config.val_freq, desc='Training')

            if step >= config.num_steps:
                print(f'\nFinished training after {step} steps\n')
                return model, step

def validate(model, val_loader, config, show_plot=False):
    model.eval()
    # model.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training 
    # and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model 
    # evaluation, and .eval() will do it for you. In addition, the common practice for evaluating/validation is using torch.no_grad() 
    # in pair with model.eval() to turn off gradients computation

    avg_val_loss = AvgMeter()
    preds = []
    targets = []
    for x, y in val_loader:
        x = x.to(config.device)
        y = y.to(config.device)

        with torch.no_grad(): # Context-manager that disabled gradient calculation
            loss, pred = model.train_step(x, y, return_prediction=True)
        avg_val_loss.add(loss.item())
        preds.append(pred.cpu())
        targets.append(y.cpu())

    # torch.cat() Concatenates the given sequence of seq tensors in the given dimension
    # All tensors must either have the same shape (except in the concatenating dimension) or be empty
    #print(preds)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    mae = mean_absolute_error(preds, targets)
    f = plot_results(preds, targets, show_plot)
    model.train()
    return {
        'val/loss': avg_val_loss.compute(),
        'val/MAE': mae,
        'val/MAE_plot': f
    }


def plot_results(preds: Tensor, targets: Tensor, show_plot: bool = False):
    # Compute the mean absolute error
    mae_test = mean_absolute_error(preds, targets)
    # Sort preds and targets to ascending targets
    sort_inds = targets.argsort() # It returns an array of indices along the given axis of the same shape as the input array, in sorted order
    targets = targets[sort_inds].numpy() # Converts a tensor object into an numpy.ndarray object
    preds = preds[sort_inds].numpy() # Converts a tensor object into an numpy.ndarray object

    f = plt.figure()
    plt.plot(targets, targets, 'r.')
    plt.plot(targets, preds, '.')
    plt.plot(targets, targets + mae_test, 'gray')
    plt.plot(targets, targets - mae_test, 'gray')
    plt.suptitle('Mean Average Error')
    plt.xlabel('True Age')
    plt.ylabel('Age predicted')
    if show_plot:
        plt.savefig('./results/age_regression_cnn/plot_results.png')
        plt.show()
        plt.close()
    return f



if __name__ == '__main__':
    # To avoid the error: OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = AgePredictionCNN().to(device)

    # Model configuration
    config, writer, optimizer, dataloaders = config_model(model)

    # Train
    model, step = train_cnn(
        config=config,
        model=model,
        optimizer=optimizer,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        writer=writer
    )

    # Validate
    val_results = validate(model=model, val_loader=dataloaders['val'], config=config, show_plot=True)
    print(val_results)
    print(f"MAE on test set: {val_results['val/MAE']:.4f}")

    # Save model
    torch.save(model.state_dict(), './models/model.pt')

    # Close tensorboard writer
    writer.close()

    # Pl

    