# import argparse
#
# import torch
# from path import Path
#
# from model import PrototypicalNetwork
# from tester import Tester
# from fire import Fire
#
#
# def test_args():
#     argparser = argparse.ArgumentParser()
#     argparser.add_argument('--checkpoint', '-c', required=True, type=Path, dest='checkpoint')
#     argparser.add_argument('--support-size', '-s', default=None, type=int)
#     argparser.add_argument('--query-size', '-q', default=None, type=int)
#     argparser.add_argument('--num-classes', '--nc', default=None, type=int)
#
#
#
# def main(dataset: str, n: int, n_s: int, n_q: int, testsize: int, use_best=False, device: str = 'cpu'):
#     """
#     Evaluate model performance into test split and get printed loss and accuracy
#     :param dataset: train dataset; can be 'omniglot' or 'miniimagenet' [str] (Note: actually works only with 'miniimagenet')
#     :param n: num classes for each meta task [int]
#     :param n_s: size of support set for each task (see paper for more details) [int]
#     :param n_q: size of query set for each task (see paper for more details) [int]
#     :param testsize: num of episodes of test split
#     :param use_best: instead of loading last epoch model parameters load the one with the lowest validation loss [bool] [default False]
#     :param device: Location of data and model parameters storage [string] [default 'cpu']
#     """
#     curr_embedding_path = EMBEDDING_PATH if not use_best else BEST_EMBEDDING_PATH
#     assert device == 'cpu' or 'cuda' in device
#     assert dataset in ['miniimagenet', 'omniglot']
#     assert curr_embedding_path.exists()
#     model = PrototypicalNetwork()
#     model.load_state_dict(torch.load(curr_embedding_path, device))
#     model = model.eval().to(device)
#     print('Loaded trained model from', curr_embedding_path)
#     if dataset == 'miniimagenet':
#         data = MiniImageNetDataLoader(1, 5, 5, n, n_s, n_q, -1, -1, testsize, device)
#     else:
#         raise NotImplementedError("Omniglot not implemented")
#     tester = Tester(model, torch.nn.CrossEntropyLoss(), device)
#     with torch.no_grad():
#         tester.test(data.test_dataloader(), testsize)
#
#
# if __name__ == '__main__':
#     Fire(main)
