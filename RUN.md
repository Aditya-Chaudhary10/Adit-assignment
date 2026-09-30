# How to run

The virtualenv lives **outside** this folder, because OneDrive locks files inside a
synced directory and breaks `pip install`.

    C:\Users\Aditya Chaudhary\.venvs\adit-voice-agent

Activate it first, in every terminal you open:

```powershell
& "$env:USERPROFILE\.venvs\adit-voice-agent\Scripts\Activate.ps1"
cd "$env:USERPROFILE\OneDrive\Desktop\Adit-assingment"
```

---

## 1. Offline demo. No phone, no browser, no worker.

Replays three scripted calls through the real analysis and Opik code.

```powershell
python demo_pipeline.py --all
```

Then open Opik and look at the three traces.
`--scenario hallucination` is the one where the agent misstates a lab value and the
judges catch it.

## 2. Talk to the agent in your browser.

Two terminals.

```powershell
# terminal 1 - the worker. Leave it running.
python agent.py dev
```

```powershell
# terminal 2 - place a simulated call
python dispatch_call.py --patient PT-10432 --simulate
python join_link.py
```

`join_link.py` prints a URL. Open it, allow the microphone, and the agent speaks first.
The room stays open 30 minutes with nobody in it; change that with `--empty-timeout`.

## 3. Call a real phone.

Needs a SIP trunk. See the README section on telephony.

```powershell
python agent.py dev                                        # terminal 1
python dispatch_call.py --patient PT-10432 --phone +91XXXXXXXXXX   # terminal 2
```

---

## Everything else

```powershell
python dispatch_call.py --list     # patients on file
python join_link.py --list         # active rooms
python -m pytest tests -q          # 41 offline tests
python agent.py console            # local microphone, no LiveKit room
python agent.py download-files     # one-off, fetches the turn-detector model
python opik_integration.py         # create server-side Opik evaluation rules
```

## After a call

| Where | What |
|---|---|
| `transcripts/<call-id>.json` | Transcript, tool calls, analysis, telemetry |
| `recordings/<call-id>.ogg` | Call audio |
| Opik project `healthcare-voice-agent` | Trace, spans, scores, audio attachment |

## If something looks wrong

Stale workers steal dispatches. If a call never connects, check for more than one:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*agent.py dev*' }
```

Kill them all and start one.
