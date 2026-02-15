#!/bin/bash
set -e

# Sweep and API key
SWEEP_ID="your_sweep_id"
WANDB_API_KEY="your_wandb_api_key"
AUTO_RESTART="true"

# Machine numbering
START=80
END=81
PREFIX="nymfe"

# Loop over machines dynamically
for i in $(seq "$START" "$END"); do
    MACHINE="${PREFIX}${i}"
    echo "🚀 Connecting to $MACHINE..."

    ssh "$MACHINE" bash -s "$SWEEP_ID" "$WANDB_API_KEY" "$AUTO_RESTART" << 'EOF'
set -e

SWEEP_ID="$1"
WANDB_API_KEY="$2"
AUTO_RESTART="$3"

cd /var/data/trebi || exit 1

echo "🔍 Diagnosing on $HOSTNAME..."

# Check if container 'uni' is running
if ! docker ps --format "table {{.Names}}" | grep -q "^uni$"; then
    echo "Container 'uni' is NOT running."

    if [ "$AUTO_RESTART" = "true" ]; then
        echo "🔄 Attempting to restart container..."

        # Ensure repository exists
        mkdir -p /var/data/xskalsky
        if [ ! -d /var/data/xskalsky/uorl ]; then
            echo "Cloning uorl repository for the first time..."
            cd /var/data/xskalsky
            git clone https://github.com/uncovsky/uorl.git
            cd uorl
            git checkout paper
            echo "🔨 Running build (first time)..."
            make build
        else
            echo "📂 uorl directory already exists, updating..."
            cd /var/data/xskalsky/uorl
            git pull
        fi

        # Restart container
        make down || true
        make up
        sleep 5

        # Run wandb agent with API key
        docker exec -d -e WANDB_API_KEY="$WANDB_API_KEY" trebi bash -c "nice -n 19 wandb agent $SWEEP_ID"
        echo "✅ Restarted container and agents."
    fi

    exit 1
fi

echo "✅ Container 'uni' is running."
EOF
    echo "✅ Finished on $MACHINE"
    echo "------------------------------"
done
