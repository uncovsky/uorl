.PHONY: build up down up-gpu

build:
	docker build . -t uorl
up-cpu:
	docker run -d \
	    --volume $(shell pwd)/ensemble_offline_rl:/work/rl/ensemble_offline_rl:z \
	    --name uorl \
	    uorl tail -f /dev/null
	docker exec -it uorl bash

up:
	docker run -d \
	    --device nvidia.com/gpu=all \
	    --volume $(shell pwd)/ensemble_offline_rl:/work/rl/ensemble_offline_rl:z \
	    --name uorl \
	    --shm-size=1g \
	    --memory=32g \
	    --pids-limit=5000 \
	    unifloral tail -f /dev/null
	docker exec -it uorl bash
down:
	docker stop uorl
	docker rm uorl
