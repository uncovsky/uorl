.PHONY: build up down up-gpu

build:
	docker build . -t unifloral
up:
	docker run -d \
	    --volume $(shell pwd)/ensemble_offline_rl:/app/ensemble_offline_rl \
	    --name uni \
	    unifloral tail -f /dev/null
	docker exec -it uni bash

attach:
	docker exec -it uni bash

up-gpu:
	docker run -d \
	    --gpus "device=0" \
	    --volume $(shell pwd)/ensemble_offline_rl:/app/ensemble_offline_rl \
	    --name uni \
	    unifloral tail -f /dev/null
	docker exec -it uni bash

down:
	docker stop uni
	docker rm uni


