from arguments import (
    DatasetArguments,
    ModelArguments, 
    TrainerArguments
)
from simple_parsing import ArgumentParser

from model.models import BaseModel

import torch

def parse_args():
    parser = ArgumentParser()
    parser.add_arguments(ModelArguments, dest="train")
    parser.add_arguments(ModelArguments, dest="valid")
    parser.add_arguments(DatasetArguments, dest="dataset")
    parser.add_arguments(TrainerArguments, dest="trainer")

    args = parser.parse_args()
    
    return args

def main():
    
    args = parse_args()

    train_args: ModelArguments     = args.train
    valid_args: ModelArguments     = args.valid
    dataset_args: DatasetArguments = args.dataset
    trainer_args: TrainerArguments = args.trainer

    model = BaseModel(train_args.num_labels)

    train_X = torch.randn(size=(10, 16))
    train_Y = torch.arange(1, 11, 1, dtype=torch.float)

    criterion = train_args.loss_fn.value
    optimizer = trainer_args.optimizer.value(model.parameters(), lr=trainer_args.lr)

    for epoch in range(trainer_args.epochs):
        out = model(train_X)
        loss = criterion(out, train_Y.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % trainer_args.print_every == 0:
            print(loss.item())

if __name__ == '__main__':
    main()