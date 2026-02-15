.PHONY: build up down up-gpu

build:
	docker build . -t unifloral
up-cpu:
	docker run -d \
	    --volume $(shell pwd)/ensemble_offline_rl:/work/rl/ensemble_offline_rl:z \
	    --name uni \
	    unifloral tail -f /dev/null
	docker exec -it uni bash

attach:
	docker exec -it uni bash

up:
	docker run -d \
	    --device nvidia.com/gpu=all \
	    --volume $(shell pwd)/ensemble_offline_rl:/work/rl/ensemble_offline_rl:z \
	    --name uni \
	    unifloral tail -f /dev/null
	docker exec -it uni bash

up-docker:
	docker run -d \
	    --gpus all \
	    --volume $(shell pwd)/ensemble_offline_rl:/work/rl/ensemble_offline_rl:z \
	    --name uni \
	    unifloral tail -f /dev/null
	docker exec -it uni bash

down:
	docker stop uni
	docker rm uni
