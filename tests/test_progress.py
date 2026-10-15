from argparse import Namespace

from vibration_analysis.progress import emit, parse


def test_desktop_progress_events_preserve_messages(capsys):
    emit(Namespace(gui_progress=True), "Test 2/2 · Point 3/5", history=False)
    event = parse(capsys.readouterr().out)
    assert event["message"] == "Test 2/2 · Point 3/5"
    assert event["history"] is False
    assert event["complete"] is False


def test_technical_and_malformed_lines_are_not_progress():
    assert parse("sensor: raw RMS=4.2 g") is None
    assert parse("@@VIBRATION_PROGRESS@@not json") is None
    assert parse('@@VIBRATION_PROGRESS@@{"message": 12}') is None


def test_cli_still_has_readable_progress(capsys):
    emit(Namespace(), "Completed", complete=True)
    assert capsys.readouterr().out == "Completed\n"
