#!/bin/bash


cd llm/demoes/llm-book
for file in **/*.ipynb; do
  if [ -f "$file" ]; then  # Check if it's a regular file
    jupyter nbconvert --Exporter.preprocessors=common.preprocess.ExtractAttachmentsPreprocessor --to notebook $file --inplace
  fi
done

jb build ../llm-book
ghp-import -n -p -f ../llm-book/_build/html
rm -rf ../llm-book/_build