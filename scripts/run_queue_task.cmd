@echo off
rem Launcher for the evaluation queue, started by Windows Task Scheduler (task "RAG_eval_queue")
rem so it runs independently of VS Code / Claude Code. Resumes from the per-question caches.
title RAG evaluation queue - do not close
cd /d "%~dp0.."
".venv\Scripts\python.exe" "scripts\run_eval_queue.py" >> "results\logs_queue_task.out" 2>&1
echo finished %date% %time% >> "results\logs_queue_task.out"
