"""
Training script for the makemore project.
"""

import os
import sys
import time
import argparse
import torch
from torch.utils.data.dataloader import DataLoader
# from torch.utils.tensorboard import SummaryWriter  # Temporarily disabled due to import issues

from config import (
    Config, ModelConfig, DataConfig, TrainingConfig,
    SamplingConfig, SystemConfig, get_default_config
)
from models import Transformer, Bigram, MLP, RNN, BoW
from data import create_datasets
from utils import print_samples, evaluate

class InfiniteDataLoader:
    """
    this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader?
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration: # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch

def main():
    # parse command line args
    parser = argparse.ArgumentParser(description="Make More")
    # system/input/output
    parser.add_argument('--input-file', '-i', type=str, help="input file with things one per line")
    parser.add_argument('--input-files', type=str, nargs='+', help="multiple input files (for Nepali names: male.txt female.txt)")
    parser.add_argument('--work-dir', '-o', type=str, help="output working directory")
    parser.add_argument('--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
    parser.add_argument('--sample-only', action='store_true', help="just sample from the model and quit, don't train")
    parser.add_argument('--num-workers', '-n', type=int, help="number of data workers for both train/test")
    parser.add_argument('--max-steps', type=int, help="max number of optimization steps to run for, or -1 for infinite.")
    parser.add_argument('--device', type=str, help="device to use for compute, examples: cpu|cuda|cuda:2|mps")
    parser.add_argument('--seed', type=int, help="seed")
    # sampling
    parser.add_argument('--top-k', type=int, help="top-k for sampling, -1 means no top-k")
    # model
    parser.add_argument('--type', type=str, help="model class type to use, bigram|mlp|rnn|gru|bow|transformer")
    parser.add_argument('--n-layer', type=int, help="number of layers")
    parser.add_argument('--n-head', type=int, help="number of heads (in a transformer)")
    parser.add_argument('--n-embd', type=int, help="number of feature channels in the model")
    parser.add_argument('--n-embd2', type=int, help="number of feature channels elsewhere in the model")
    # optimization
    parser.add_argument('--batch-size', '-b', type=int, help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', type=float, help="learning rate")
    parser.add_argument('--weight-decay', '-w', type=float, help="weight decay")
    args = parser.parse_args()

    # Load default configuration
    config = get_default_config()

    # Override config with command line arguments
    if args.input_file:
        config.data.input_file = args.input_file
    if args.input_files:
        config.data.input_files = args.input_files
    if args.work_dir:
        config.system.work_dir = args.work_dir
    if args.resume:
        config.system.resume = args.resume
    if args.sample_only:
        config.system.sample_only = args.sample_only
    if args.num_workers:
        config.system.num_workers = args.num_workers
    if args.max_steps:
        config.training.max_steps = args.max_steps
    if args.device:
        config.system.device = args.device
    if args.seed:
        config.system.seed = args.seed
    if args.top_k:
        config.sampling.top_k = args.top_k
    if args.type:
        config.system.model_type = args.type
    if args.n_layer:
        config.model.n_layer = args.n_layer
    if args.n_head:
        config.model.n_head = args.n_head
    if args.n_embd:
        config.model.n_embd = args.n_embd
    if args.n_embd2:
        config.model.n_embd2 = args.n_embd2
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.weight_decay:
        config.training.weight_decay = args.weight_decay

    print("Configuration:")
    print(f"  Data: {config.data}")
    print(f"  Model: {config.model}")
    print(f"  Training: {config.training}")
    print(f"  Sampling: {config.sampling}")
    print(f"  System: {config.system}")

    # system inits
    torch.manual_seed(config.system.seed)
    torch.cuda.manual_seed_all(config.system.seed)
    os.makedirs(config.system.work_dir, exist_ok=True)
    # writer = SummaryWriter(log_dir=config.system.work_dir)  # Temporarily disabled

    # init datasets
    input_files = config.data.input_files if config.data.input_files else config.data.input_file
    train_dataset, test_dataset = create_datasets(input_files, config.data, use_lowercase=True)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    print(f"dataset determined that: {vocab_size=}, {block_size=}")

    # init model
    config.model.vocab_size = vocab_size
    config.model.block_size = block_size
    if config.system.model_type == 'transformer':
        model = Transformer(config.model)
    elif config.system.model_type == 'bigram':
        model = Bigram(config.model)
    elif config.system.model_type == 'mlp':
        model = MLP(config.model)
    elif config.system.model_type == 'rnn':
        model = RNN(config.model, cell_type='rnn')
    elif config.system.model_type == 'gru':
        model = RNN(config.model, cell_type='gru')
    elif config.system.model_type == 'bow':
        model = BoW(config.model)
    else:
        raise ValueError(f'model type {config.system.model_type} is not recognized')
    model.to(config.system.device)
    print(f"model #params: {sum(p.numel() for p in model.parameters())}")
    if config.system.resume or config.system.sample_only: # note: if we sample-only then we also assume we are resuming
        print("resuming from existing model in the workdir")
        model.load_state_dict(torch.load(os.path.join(config.system.work_dir, 'model.pt')))
    if config.system.sample_only:
        print_samples(num=config.sampling.num_samples, model=model, train_dataset=train_dataset, test_dataset=test_dataset, config=config)
        sys.exit()

    # init optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=config.training.betas,
        eps=config.training.eps
    )

    # init dataloader
    batch_loader = InfiniteDataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        pin_memory=True,
        num_workers=config.system.num_workers
    )

    # training loop
    best_loss = None
    step = 0
    while True:

        t0 = time.time()

        # get the next batch, ship to device, and unpack it to input and target
        batch = batch_loader.next()
        batch = [t.to(config.system.device) for t in batch]
        X, Y = batch

        # feed into the model
        logits, loss = model(X, Y)

        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        if config.training.gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
        optimizer.step()

        # wait for all CUDA work on the GPU to finish then calculate iteration time taken
        if config.system.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()

        # logging
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")

        # evaluate the model
        if step > 0 and step % config.training.eval_interval == 0:
            train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10, device=config.system.device)
            test_loss  = evaluate(model, test_dataset,  batch_size=100, max_batches=10, device=config.system.device)
            # writer.add_scalar("Loss/train", train_loss, step)  # Temporarily disabled
            # writer.add_scalar("Loss/test", test_loss, step)   # Temporarily disabled
            # writer.flush()  # Temporarily disabled
            print(f"step {step} train loss: {train_loss} test loss: {test_loss}")
            # save the model to disk if it has improved
            if best_loss is None or test_loss < best_loss:
                out_path = os.path.join(config.system.work_dir, "model.pt")
                print(f"test loss {test_loss} is the best so far, saving model to {out_path}")
                torch.save(model.state_dict(), out_path)
                best_loss = test_loss

        # sample from the model
        if step > 0 and step % config.training.sample_interval == 0:
            print_samples(num=config.sampling.num_samples, model=model, train_dataset=train_dataset, test_dataset=test_dataset, config=config)

        step += 1
        # termination conditions
        if config.training.max_steps >= 0 and step >= config.training.max_steps:
            break

if __name__ == '__main__':
    main()
