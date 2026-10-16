import importlib.util
from pathlib import Path
import shutil
import sys
import unittest
import uuid
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('schedule', REPO / 'notebooks/molab_eval_schedule.py')
schedule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schedule)


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.root = REPO / 'tmp' / ('schedule_qa_' + uuid.uuid4().hex)
        executable = self.root / '.venv' / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
        executable.parent.mkdir(parents=True)
        executable.write_bytes(b'')

    def tearDown(self):
        shutil.rmtree(self.root)

    def run_schedule(self, extra=(), codes=(0, 0)):
        commands = []
        return_codes = iter(codes)

        class Child:
            def __init__(self, command):
                commands.append(command)
                self.code = next(return_codes)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def wait(self):
                return self.code

            def poll(self):
                return self.code

        argv = ['schedule', 'run', '--root', str(self.root), '--max-hours', '2', '--batch-size', '32', *extra]
        with patch.object(sys, 'argv', argv), patch.object(schedule.subprocess, 'Popen', Child):
            code = schedule.main()
        return code, commands

    def test_captions_before_judge_and_forward_batch(self):
        code, commands = self.run_schedule()
        self.assertEqual(code, 0)
        self.assertEqual(len(commands), 2)
        self.assertIn('--captions-only', commands[0])
        self.assertNotIn('--captions-only', commands[1])
        self.assertEqual(commands[1][commands[1].index('--batch-size') + 1], '32')

    def test_pause_stops_before_judge(self):
        code, commands = self.run_schedule(codes=(75,))
        self.assertEqual(code, 75)
        self.assertEqual(len(commands), 1)

    def test_captions_only_does_not_start_judge(self):
        code, commands = self.run_schedule(extra=('--captions-only',), codes=(0,))
        self.assertEqual(code, 0)
        self.assertEqual(len(commands), 1)


if __name__ == '__main__':
    unittest.main()
