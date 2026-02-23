# DIFE CLI for Grok Ecosystem

[![CI](https://github.com/AdemVessell/dife-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/AdemVessell/dife-cli/actions)
[![PyPI](https://img.shields.io/pypi/v/dife-cli)](https://pypi.org/project/dife-cli/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository provides a **production-ready CLI tool** for the DIFE (Decaying Information Forgetting Equation) framework, optimized for continual learning in xAI’s Grok ecosystem.

This repository provides a **production-ready CLI tool** for the DIFE (Decaying Information Forgetting Equation) framework, optimized for continual learning in xAI’s Grok ecosystem.

DIFE models information decay over sequential tasks using:

\[ Q_n = \max(0, Q_0 \cdot \alpha^n - \beta \cdot n \cdot (1 - \alpha^n)) \]

**DIFE directly tackles catastrophic forgetting by dynamically scheduling replay strength — protecting important old knowledge early while intelligently reducing rehearsal as new tasks accumulate.**

Built live with **Grok 4.20 Heavy (16-agent swarm)**. Designed to advance Grok’s memory retention and adaptation.

## Installation
```bash
git clone https://github.com/AdemVessell/dife-cli.git
cd dife-cli
pip install -e .
