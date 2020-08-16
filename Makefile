clean-logs:
	rm -r logs/
clean-models:
	rm -r model_weights
clean-all: clean-logs clean-models

train-miniimagenet:
	python3 train.py --dataset='miniimagenet' --train-n=30 --val-n=5 --test-n=5 --n-s=1 --n-q=15 --trainsize=1_000 --valsize=64 --testsize=64 --device='cuda:1' --batch-size=1

train-miniimagenet-log:
	python3 train.py --dataset='miniimagenet' --epochs=100 --train-n=30 --val-n=5 --test-n=5 --n-s=1 --n-q=15 --trainsize=1_000 --valsize=64 --testsize=64 --device='cuda:1' --batch-size=1 > train-log.txt 2> train-err.txt

test-miniimagenet:
    python3 test.py --dataset='miniimagenet' --n=5 --n-s=1 --n-q=15 --testsize=10000 --device='cuda:0'
