uv run python cs336_systems/benchmarking_script.py --d_model 768 --num_layers 12 --num_heads 12 --d_ff 3072

uv run python cs336_systems/benchmarking_script.py --d_model 1024 --num_layers 24 --num_heads 16 --d_ff 4096

uv run python cs336_systems/benchmarking_script.py --d_model 1280 --num_layers 36 --num_heads 20 --d_ff 5120

uv run python cs336_systems/benchmarking_script.py --d_model 1600 --num_layers 48 --num_heads 25 --d_ff 6400

uv run python cs336_systems/benchmarking_script.py --d_model 2560 --num_layers 32 --num_heads 32 --d_ff 10240
