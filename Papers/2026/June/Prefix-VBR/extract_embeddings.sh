#!/bin/bash
# extract_embeddings.sh — Extract granite embeddings from PostgreSQL
#
# Usage: ./extract_embeddings.sh [output_file]
#
# Requires: psql access to claude_conversations database
# with wordnet_embeddings_granite table.

OUTPUT="${1:-embeddings_full.csv}"

echo "Extracting granite embeddings to ${OUTPUT}..."
psql -d claude_conversations -t -A -c \
    "SELECT replace(replace(embedding::text,'[',''),']','')
     FROM wordnet_embeddings_granite" > "${OUTPUT}"

LINES=$(wc -l < "${OUTPUT}")
VALUES=$((LINES * 384))
echo "Extracted ${LINES} embeddings (${VALUES} values) to ${OUTPUT}"
