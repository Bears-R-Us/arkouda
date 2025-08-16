#!/bin/bash
# scripts/trim_server_modules.sh

CFG_FILE="ServerModules.cfg"

# List of modules to exclude in multidim builds
EXCLUDE_MODULES=(
  "CheckpointMsg"
  "CommDiagnosticsMsg"
  "ArraySetops"
  "ArraySetopsMsg"
  "DataFrameIndexingMsg" 
  "EncodingMsg"
  "Codecs"
  "CSVMsg"
  "ExternalIntegration"
)

# Backup original
cp "$CFG_FILE" "$CFG_FILE.bak"

# Comment out each module if present
for module in "${EXCLUDE_MODULES[@]}"; do
  sed -i "s/^\($module\)$/# \1/" "$CFG_FILE"
done


