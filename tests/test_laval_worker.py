"""Prove that a dirty notebook kernel cannot contaminate scientific workers."""
import ast, contextlib, csv, io, json, os, sys, tempfile, types, unittest, zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import colab_environment as environment
ROOT = Path(__file__).resolve().parents[1]

class WorkerTests(unittest.TestCase):
    def test_worker_mode_ignores_loaded_library_versions(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"requirements.txt"
            path.write_text("numpy==2.2.6\n")
            pins = {"numpy":"2.2.6"}
            with patch.object(environment.sys,"version_info",(3,13)), \
                 patch.object(environment,"installed_versions",return_value=pins), \
                 patch.object(environment,"loaded_versions",side_effect=AssertionError("inspected parent")), \
                 patch.object(environment,"validate_kernel",side_effect=AssertionError("imported parent NumPy")), \
                 patch.object(environment,"probe_fresh_process",return_value=SimpleNamespace(returncode=0,stdout="worker passed")):
                self.assertEqual(environment.ensure_environment(path,validate_active_kernel=False),pins)

    def test_notebook_has_no_scientific_imports(self):
        nb=json.loads((ROOT/"notebooks/retrain_laval.ipynb").read_text(encoding="utf-8"))
        banned={"numpy","pandas","sklearn","scipy","matplotlib","joblib","laval_training"}
        for cell in nb["cells"]:
            if cell["cell_type"] != "code": continue
            for node in ast.walk(ast.parse("".join(cell["source"]))):
                names=[]
                if isinstance(node,ast.Import): names=[a.name for a in node.names]
                elif isinstance(node,ast.ImportFrom): names=[node.module or ""]
                self.assertFalse({n.split(".")[0] for n in names} & banned)
        self.assertEqual(nb["metadata"]["utsm_execution_mode"],"fresh-python-worker")

    def test_entire_notebook_with_poisoned_parent_libraries(self):
        pins=environment.parse_pins((ROOT/"requirements-colab.txt").read_text())
        if sys.version_info[:2] < (3,10) or environment.installed_versions(pins) != pins:
            self.skipTest("requires supported Python with exact pinned packages")
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp); archive=temp/"telemetry.zip"
            with zipfile.ZipFile(archive,"w") as z:
                for session in range(3):
                    stream=io.StringIO(); writer=csv.writer(stream)
                    writer.writerow(["timestamp_ms","current_mA","voltage_mV","wheel_speed_valid","wheel_speed_kmph","temperature_C","temperature_valid"])
                    for i in range(120): writer.writerow([i*500,-3000-i,24289,1,20+(i%8)/10+session,25,1])
                    z.writestr(f"2026-09-{19 if session<2 else 20}-{session}.csv",stream.getvalue())
            poisoned={}
            for name in environment.MODULES.values():
                module=types.ModuleType(name); module.__version__="incompatible-kernel-version"
                def fail(attribute): raise AssertionError("Kernel tried to use a scientific library")
                module.__getattr__=fail; poisoned[name]=module
            nb=json.loads((ROOT/"notebooks/retrain_laval.ipynb").read_text(encoding="utf-8"))
            variables={"UTSM_ARCHIVE_PATH":str(archive),"UTSM_WORK_DIR":str(temp/"workspace"),
                       "UTSM_OUTPUT_DIR":str(temp/"results"),"UTSM_N_ESTIMATORS":"2"}
            displays=[]
            def raw_display(payload, raw=False):
                self.assertTrue(raw)
                self.assertIsInstance(payload["text/html"],str)
                displays.append(payload["text/html"])
            namespace={"__name__":"__main__", "display":raw_display}
            with patch.dict(os.environ,variables), patch.dict(sys.modules,poisoned), contextlib.redirect_stdout(io.StringIO()):
                for i,cell in enumerate(nb["cells"]):
                    if cell["cell_type"]=="code": exec(compile("".join(cell["source"]),f"cell-{i}","exec"),namespace)
                for name,module in poisoned.items(): self.assertIs(sys.modules[name],module)
            self.assertTrue(any("data:image/png;base64" in html for html in displays))
            output=temp/"results"
            self.assertTrue((output/"random_forest_development.joblib").is_file())
            self.assertTrue(json.loads((output/"inference_example.json").read_text())["passed"])
            with zipfile.ZipFile(str(output)+".zip") as z: self.assertIn("inference_example.json",z.namelist())

if __name__=="__main__": unittest.main()
