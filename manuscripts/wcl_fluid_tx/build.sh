#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TEXINPUTS="./vendor/:${TEXINPUTS:-}"
for pass in 1 2; do
  pdflatex -interaction=nonstopmode -halt-on-error main.tex > "build_main_${pass}.log"
  pdflatex -interaction=nonstopmode -halt-on-error -jobname=main_single_column '\def\SingleColumn{1}\input{main.tex}' > "build_single_${pass}.log"
  pdflatex -interaction=nonstopmode -halt-on-error initial_model_user.tex > "build_model_${pass}.log"
done
