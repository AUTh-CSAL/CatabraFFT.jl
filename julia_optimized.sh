#!/bin/bash
# Save as: ~/julia-optimized.sh
# Make executable: chmod +x ~/julia-optimized.sh

exec julia --cpu-target=native -O3 --inline=yes --check-bounds=no --project=@. "$@"
