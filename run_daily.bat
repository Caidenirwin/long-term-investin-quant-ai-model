@echo off
rem Daily run for Windows Task Scheduler. Logs go to live\run_log.txt.
rem Forward test (mode = "sim"): schedule after the close, e.g. 5:00 pm.
rem IBKR (paper or live): schedule for about 3:00 pm Toronto time so market-on-close orders make the cutoff.
cd /d "%~dp0"
if not exist live mkdir live
echo ==== %date% %time% ==== >> live\run_log.txt
python live_trader.py >> live\run_log.txt 2>&1
rem The AI's fake-money account (always simulated, so it runs after the close like the rules forward test).
echo ---- AI forward test ---- >> live\run_log.txt
python -m ai_investor.forward >> live\run_log.txt 2>&1
