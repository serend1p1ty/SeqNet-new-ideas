#!/bin/bash -e

# Run this script at project root by "./dev/linter.sh" before you commit

{
  black --version | grep -E "21.9b0" > /dev/null
} || {
  echo "Linter requires 'black==21.9b0' !"
  exit 1
}

ISORT_VERSION=$(isort --version-number)
if [[ "$ISORT_VERSION" != 5.9.3 ]]; then
  echo "Linter requires isort==5.9.3 !"
  exit 1
fi

echo "Running isort ..."
isort --line-length=100 --profile=black .

echo "Running black ..."
black --line-length=100 .

echo "Running flake8 ..."
if [ -x "$(command -v flake8-3)" ]; then
  flake8-3 .
else
  python3 -m flake8 .
fi

command -v arc > /dev/null && arc lint