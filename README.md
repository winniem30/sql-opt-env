---
title: SQL Opt Env
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_file: server.py
pinned: false
---

# SQL Optimization Environment (OpenEnv)

This project implements an OpenEnv-compatible environment for optimizing SQL queries.

## 🚀 Features
- 3 tasks (easy → medium → hard)
- Deterministic graders
- Reward-based optimization
- Flask API endpoints: `/reset`, `/step`, `/state`

## 📡 API Usage

### Reset
POST /reset

### Step
POST /step
{
  "query": "SELECT id FROM users"
}

## 🐳 Deployment
Deployed using Docker on Hugging Face Spaces.

## 🏁 Status
Ready for Meta OpenEnv Hackathon submission.