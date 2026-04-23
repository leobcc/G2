```
CONFIDENTIAL — For candidate use only. Do not distribute.
```
```
Groupon — Global Operations
```
# Content That Converts

## Context

Groupon is rebuilding deal creation on a structured platform. Every deal is a typed object with fields:
title, description, fine print, category, geo, pricing options, image quality score. Today roughly 100
FTE write, vet, and edit this content by hand. The team optimizes for throughput — deals ship on
time. Nobody measures whether the content actually converts. There is no feedback loop between
what gets written and what gets bought.

You have been given a dataset of 500 deals with their content fields and 8 weeks of performance
data. Your job: figure out what separates high-converting content from low-converting content, build
a system that scores and rewrites underperformers, and show how this replaces manual content
operations at scale.

## Deliverables

### 1. Analysis

What patterns in the content data correlate with conversion performance? Be specific. Show your
work.

### 2. Working Proof of Concept

A system that takes a low-converting deal's structured fields and its performance data, scores the
content, and produces an improved version. The system must include an evaluation mechanism
that measures whether rewrites are actually better than the originals.

Submit as a runnable repository.

### 3. Operations Blueprint

One to two pages. How does this system scale to handle all deals? What remains human? What is
the path from 100 FTE to target state? What do you measure weekly to know it is working?

## What We Provide

- deals.csv — 500 deals with content fields and conversion data
- data_dictionary.md — field descriptions

## What We Evaluate

All criteria carry equal weight.

- Data instinct — Did you find what matters in the data?
- System thinking — Does your system get better over time, or is it a one-shot prompt?


- Working code — Does it run?
- Eval rigor — How do you know the output is better?
- Automation mindset — Are you replacing manual work, or just augmenting it?

## Submission

A git repository (GitHub link or zip) containing your analysis, code, and operations blueprint.


