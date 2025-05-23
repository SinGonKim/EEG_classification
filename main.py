from config import arg
from data_loader import data_loader
from build_net import build_net
from make_solver import make_solver
from utils import control_random, timeit
import os


@timeit
def main():
    args = arg() # config 파일에 존재
    # seed control
    if args.seed:
        control_random(args) # 쿠다, 추론 속도 등 조절

    # load train / test dataset
    train_loader, val_loader = data_loader(args)

    # import backbone model
    net = build_net(args, train_loader.dataset.X.shape)

    # make solver (runner)
    solver = make_solver(args, net, train_loader, val_loader)

    # train
    solver.experiment()


if __name__ == '__main__':
    main()
