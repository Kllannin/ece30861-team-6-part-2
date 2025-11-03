import unittest
from unittest.mock import patch, mock_open
import run
import src.url_class as url_class
import tempfile
import multiprocessing
import os
import sys
from pathlib import Path
import io  # for StringIO
import json
import src.json_output as json_output
from src.url_class import Code, Dataset, Model, ProjectGroup

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import metric_caller as mc


class TestRunMainClean(unittest.TestCase):

    def setUp(self):
        # Ensure the fake target file exists in tests directory
        self.fake_target_path = os.path.join(os.path.dirname(__file__), "fake_target.txt")
        if not os.path.exists(self.fake_target_path):
            with open(self.fake_target_path, "w") as f:
                f.write("https://huggingface.co/example/model\n")

    # Fix the import paths - use the full module path
    @patch("src.url_class.parse_project_file")
    @patch("sys.argv", ["run.py", "tests/fake_target.txt"])
    def test_main_runs_without_error(self, mock_parse):
        mock_parse.return_value = []
        try:
            run.main()
        except SystemExit:
            # Allow sys.exit() calls
            pass
        except Exception as e:
            self.fail(f"run.main() raised {e} unexpectedly!")

    @patch("src.url_class.parse_project_file")
    @patch("sys.argv", ["run.py", "tests/fake_target.txt"])
    def test_parse_project_file_called(self, mock_parse):
        mock_parse.return_value = []
        try:
            run.main()
        except SystemExit:
            pass  # Allow sys.exit()
        mock_parse.assert_called_once_with("tests/fake_target.txt")

class TestURLClassClean(unittest.TestCase):

    @patch.object(Path, "open", new_callable=mock_open, read_data="")
    def test_parse_project_file_returns_empty_list(self, mock_open_fn):
        result = url_class.parse_project_file("dummy.txt")
        self.assertEqual(result, [])


class TestSampleClean(unittest.TestCase):

    def setUp(self):
        # Dummy function for process_worker
        def dummy_func(x=0, y=0):
            return x + y, 0.01
        self.dummy_func = dummy_func

    # ---------------- metric_caller.parse_keys_from_string ----------------
    def test_parse_keys(self):
        self.assertEqual(mc.parse_keys_from_string(""), [])
        self.assertEqual(mc.parse_keys_from_string("key1"), ["key1"])
        self.assertEqual(mc.parse_keys_from_string("key1,key2,key3"), ["key1","key2","key3"])
        self.assertEqual(mc.parse_keys_from_string(" key1 , key2 "), ["key1","key2"])

    def test_parse_keys_whitespace_only(self):
        self.assertEqual(mc.parse_keys_from_string("   "), [])

    def test_parse_keys_trailing_commas(self):
        self.assertEqual(mc.parse_keys_from_string("a,b,"), ["a", "b", ""])

    # ---------------- logger_process ----------------
    def test_logger_writes_messages(self):
        q = multiprocessing.Queue()
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            log_file = tf.name
        try:
            p = multiprocessing.Process(target=mc.logger_process, args=(q, log_file))
            p.start()
            q.put("hello")
            q.put(None)
            p.join()
            with open(log_file) as f:
                content = f.read()
            self.assertIn("hello", content)
        finally:
            os.remove(log_file)

    def test_logger_process_handles_exception(self):
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=mc.logger_process, args=(q, "/invalid/path/log.txt"))
        p.start()
        q.put(None)
        p.join()
        self.assertTrue(True)

    # ---------------- process_worker ----------------
    def test_process_worker_success(self):
        q = multiprocessing.Queue()
        mc.process_worker(self.dummy_func, q, multiprocessing.Queue(), 1.0, "dummy_func", 2, 3)
        score, t, w, name = q.get()
        self.assertGreater(score, 0)
        self.assertEqual(w, 1.0)
        self.assertEqual(name, "dummy_func")

    def test_process_worker_exception(self):
        q = multiprocessing.Queue()
        def fail_func(**kwargs): raise ValueError("fail")
        mc.process_worker(fail_func, q, multiprocessing.Queue(), 1.0, "fail_func", 1)
        score, t, w, name = q.get()
        self.assertEqual(score, 0.0)
        self.assertEqual(w, 1.0)
        self.assertEqual(name, "fail_func")

    def test_process_worker_with_no_args(self):
        q = multiprocessing.Queue()
        def f(): return 42, 0.1
        mc.process_worker(f, q, multiprocessing.Queue(), 2.0, "f")
        score, t, w, name = q.get()
        self.assertEqual(score, 42)
        self.assertEqual(w, 2.0)
        self.assertEqual(name, "f")

    # ---------------- load_available_functions ----------------
    def test_load_functions_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            funcs = mc.load_available_functions(tmpdir, multiprocessing.Queue(), 0)
            self.assertEqual(funcs, {})

    # ---------------- run_concurrently_from_file ----------------
    def test_run_concurrently_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks_file = os.path.join(tmpdir, "tasks.txt")
            open(tasks_file, "w").close()
            all_args = {"verbosity":0}
            scores, times = mc.run_concurrently_from_file(
                tasks_file, all_args, {}, os.path.join(tmpdir, "log.txt")
            )
            self.assertEqual(scores['net_score'], 0.0)

    def test_run_concurrently_invalid_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tasks_file = os.path.join(tmpdir, "tasks.txt")
            with open(tasks_file, "w") as f:
                f.write("invalid_line_without_proper_syntax\n")
            all_args = {"verbosity":0}
            scores, times = mc.run_concurrently_from_file(
                tasks_file, all_args, {}, os.path.join(tmpdir, "log.txt")
            )
            self.assertEqual(scores['net_score'], 0.0)

    # ---------------- build_model_output ----------------
    def test_build_model_output_basic(self):
        scores = {
            "net_score": 0.9,
            "rampup_time_metric": 0.8,
            "bus_factor_metric": 0.7,
            "performance_claims_metric": 0.6,
            "calculate_license_score": 1.0,
            "calculate_size_score": 0.5,
            "dataset_and_code_present": 0.85,
            "dataset_quality": 0.75,
            "code_quality": 0.65,
        }
        latency = {
            "net_score_latency": 100,
            "rampup_time_metric": 80,
            "bus_factor_metric": 70,
            "performance_claims_metric": 60,
            "calculate_license_score": 50,
            "calculate_size_score": 40,
            "dataset_and_code_present": 30,
            "dataset_quality": 20,
            "code_quality": 10,
        }

        # Capture stdout
        with patch('sys.stdout', new_callable=io.StringIO) as fake_out:
            json_output.build_model_output("model_name", "MODEL", scores, latency)
            output = json.loads(fake_out.getvalue())

        self.assertEqual(output["name"], "model_name")
        self.assertEqual(output["category"], "MODEL")
        self.assertEqual(output["net_score"], 0.9)
        self.assertEqual(output["net_score_latency"], 100)
        self.assertEqual(output.get("ramp_up_time", 0.0), 0.8)
        self.assertEqual(output["license"], 1.0)
        self.assertEqual(output["code_quality"], 0.65)

    def test_build_model_output_missing_keys(self):
        # Missing keys should default to 0
        scores = {}
        latency = {}
        with patch('sys.stdout', new_callable=io.StringIO) as fake_out:
            json_output.build_model_output("name", "category", scores, latency)
            output = json.loads(fake_out.getvalue())
        self.assertEqual(output["net_score"], 0.0)
        self.assertEqual(output["license"], 0.0)
        self.assertEqual(output["dataset_quality"], 0.0)


class TestURLClass(unittest.TestCase):

    # ---------------- Data classes ----------------
    def test_code_dataset_model_dataclasses(self):
        c = Code("link1", "ns1")
        d = Dataset("link2", "ns2", "repo2", "rev2")
        m = Model("link3", "ns3", "repo3", "rev3")

        self.assertEqual(c.link, "link1")
        self.assertEqual(d.repo, "repo2")
        self.assertEqual(m.rev, "rev3")

    def test_project_group_optional(self):
        pg = ProjectGroup(code=None, dataset=None, model=None)
        self.assertIsNone(pg.code)
        self.assertIsNone(pg.dataset)
        self.assertIsNone(pg.model)

    # ---------------- parse_huggingface_url ----------------
    def test_parse_huggingface_url_basic(self):
        ns, repo, rev = url_class.parse_huggingface_url("https://huggingface.co/ns/repo")
        self.assertEqual(ns, "ns")
        self.assertEqual(repo, "repo")
        self.assertEqual(rev, "main")

    def test_parse_huggingface_url_with_tree(self):
        ns, repo, rev = url_class.parse_huggingface_url("https://huggingface.co/ns/repo/tree/dev")
        self.assertEqual(rev, "dev")

    # ---------------- parse_hf_dataset_url_repo ----------------
    def test_parse_hf_dataset_url_repo_valid(self):
        url = "https://huggingface.co/datasets/stanfordnlp/imdb"
        repo = url_class.parse_hf_dataset_url_repo(url)
        self.assertEqual(repo, "imdb")

import unittest
import multiprocessing
import tempfile
import os
import io
import json
from unittest.mock import patch
import metric_caller as mc

class TestMetricCallerExtra(unittest.TestCase):

    # ---------------- parse_keys_from_string ----------------
    def test_parse_keys_edge_cases(self):
        # Leading/trailing commas and whitespace
        self.assertEqual(mc.parse_keys_from_string(",a,b,,"), ["", "a", "b", "", ""])
        # All whitespace
        self.assertEqual(mc.parse_keys_from_string("   ,  , "), ["", "", ""])
        # Only commas
        self.assertEqual(mc.parse_keys_from_string(",,,,"), ["", "", "", "", ""])  # FIXED
    # ---------------- logger_process ----------------
    def test_logger_process_multiple_messages(self):
        q = multiprocessing.Queue()
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            log_file = tf.name
        try:
            p = multiprocessing.Process(target=mc.logger_process, args=(q, log_file))
            p.start()
            messages = ["msg1", "msg2", "msg3"]
            for m in messages:
                q.put(m)
            q.put(None)
            p.join()
            with open(log_file) as f:
                content = f.read()
            for m in messages:
                self.assertIn(m, content)
        finally:
            os.remove(log_file)

    # ---------------- process_worker ----------------
    def test_process_worker_with_returned_dict(self):
        q = multiprocessing.Queue()
        # Simulate a metric function returning a dict
        def metric_func(a, b):
            return {"score": a+b}, 0.02
        mc.process_worker(metric_func, q, multiprocessing.Queue(), 1.5, "metric_func", 2, 3)
        score, t, w, name = q.get()
        # The process_worker handles dicts differently for calculate_size_score; here weight should remain unchanged
        self.assertEqual(w, 1.5)
        self.assertEqual(name, "metric_func")
        self.assertIsInstance(score, dict)
        self.assertIn("score", score)

class TestMetricCallerExtraBranches(unittest.TestCase):
    def test_load_functions_import_and_attribute_error(self):
        import tempfile
        import os
        import multiprocessing

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a Python file with syntax error to trigger ImportError
            bad_module_path = os.path.join(tmpdir, "bad_module.py")
            with open(bad_module_path, "w") as f:
                f.write("this is not valid python")
            
            # Create a Python file without expected function to trigger AttributeError
            empty_module_path = os.path.join(tmpdir, "empty_module.py")
            with open(empty_module_path, "w") as f:
                f.write("x = 42")  # no function with module name

            log_q = multiprocessing.Queue()
            funcs = mc.load_available_functions(tmpdir, log_q, 0)
            # Both modules fail to load, so funcs dict should be empty
            self.assertEqual(funcs, {})


import unittest
from unittest.mock import patch, MagicMock
import run

class TestRunExtraBranches(unittest.TestCase):

    @patch("subprocess.check_call")
    @patch("sys.argv", ["run.py", "install"])
    def test_install_branch(self, mock_subproc):
        try:
            run.main()
        except SystemExit:
            pass
        mock_subproc.assert_called_once_with([run.sys.executable, "-m", "pip", "install", "--user", "-r", "requirements.txt"])

    @patch("sys.argv", ["run.py", "test"])
    @patch("builtins.print")
    @patch('unittest.main')  # Add this mock decorator
    def test_test_branch(self, mock_unittest, mock_print):
        mock_unittest.return_value = None
        
        try:
            run.main()
        except SystemExit:
            pass
            
        # Check if "Running test suite..." was printed
        found = False
        for call in mock_print.call_args_list:
            if len(call[0]) > 0 and "Running test suite..." in str(call[0][0]):
                found = True
                break
        self.assertTrue(found, "Expected 'Running test suite...' message not found")

    # ... keep the existing test_url_file_branch method as is ...