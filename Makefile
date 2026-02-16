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
	    --shm-size=1g \
	    --memory=32g \
	    --pids-limit=5000 \
	    unifloral tail -f /dev/null
	docker exec -it uni bash

up-docker:
	docker run -d \
	    --gpus all \
	    --volume $(shell pwd)/ensemble_offline_rl:/work/rl/ensemble_offline_rl:z \
	    --name uni \
	    --shm-size=1g \
	    --memory=32g \
	    --pids-limit=5000 \
	    unifloral tail -f /dev/null
	docker exec -it uni bash

up-arachne:
	docker run -d \
	  --volume $(shell pwd)/ensemble_offline_rl:/work/rl/ensemble_offline_rl:z \
	  --device nvidia.com/gpu=all \
	  --group-add keep-groups \
	  --security-opt label=disable \
	  --shm-size=1g \
	  --memory=32g \
	  --pids-limit=5000 \
	  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
	  -e CUDA_VISIBLE_DEVICES=2,3 \
	  --name uni \
	  unifloral tail -f /dev/null

down:
	docker stop uni
	docker rm uni
