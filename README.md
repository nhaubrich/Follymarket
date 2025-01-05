# Follymarket: Analyzing Polymarket's Successes and Failures
This repo contains python-based tools to analyze Polymarkets's prediction ability.
In `scraper.py`, market and price data are fetched from Polymarket's historical "gamma" API.
In `analyze.py`, several metrics and plots are computed, including the decomposed Brier score and calibration. Errors for calibration plots are computed through a quantile bootstrap.