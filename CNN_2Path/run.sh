#!/bin/bash

for f in $(ls extracted_features)
do
    full_path="extracted_features/$f"
    echo $full_path
done
