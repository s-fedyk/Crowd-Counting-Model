build:
	python3 ./src/main.py
load:
	python3 ./src/main.py --checkpoint-path="./experiments/best_checkpoint.pth.tar"
eval:
	python3 ./src/eval.py
packages:
	python3 -m pip install -r ./requirements.txt
tensorboard:
	python3 -m tensorboard.main --logdir=runs
shtech:
	python3 src/shtech.py
clean:
	rm -rf processed/*
