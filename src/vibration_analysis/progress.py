"""Small progress events for the desktop front end; technical stdout stays available."""

import json

PREFIX = "@@VIBRATION_PROGRESS@@"


def emit(args, message, *, history=True, complete=False):
    if getattr(args, "gui_progress", False):
        print(
            PREFIX + json.dumps({"message": message, "history": history, "complete": complete}),
            flush=True,
        )
    else:
        print(message, flush=True)


def parse(line):
    if not line.startswith(PREFIX):
        return None
    try:
        event = json.loads(line[len(PREFIX) :])
    except (ValueError, TypeError):
        return None
    if not isinstance(event, dict) or not isinstance(event.get("message"), str):
        return None
    return event
