clean-logs:
	rm -r lightning_logs/
clean-models:
	rm -r model_weights


train-miniimagenet:
	python3 train.py --dataset='miniimagenet' --train-n=15 --val-n=5 --test-n=5 --n-s=1 --n-q=15 --trainsize=1_000 --valsize=512 --testsize=512 --gpu=[1] --batch-size=1

train-miniimagenet-log-err:
	python3 train.py --dataset='miniimagenet' --train-n=5 --val-n=5 --test-n=5 --n-s=1 --n-q=15 --trainsize=1_000 --valsize=512 --testsize=512 --gpu=[1] --batch-size=4 2> train-err.txt
