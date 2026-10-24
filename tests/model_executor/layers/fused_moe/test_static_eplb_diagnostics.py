# SPDX-License-Identifier: Apache-2.0
"""CPU-only lifecycle contracts; no torch/vLLM imports or GPU execution."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[4] / "vllm_hcu/distributed/static_eplb_diagnostics.py"


class Tensor:
    """CPU storage adapter for framework counters; never a GPU/mapper oracle."""
    def __init__(self, values):self.values = np.asarray(values)
    @property
    def shape(self):return self.values.shape
    @property
    def ndim(self):return self.values.ndim
    @property
    def dtype(self):return "torch.int64" if self.values.dtype == np.int64 else "torch.int32"
    @property
    def device(self):return SimpleNamespace(type="cuda")
    def data_ptr(self):return self.values.ctypes.data
    def numel(self):return self.values.size
    def is_contiguous(self):return self.values.flags.c_contiguous
    def zero_(self):self.values.fill(0);return self
    def fill_(self, value):self.values.fill(value);return self
    def copy_(self, other):np.copyto(self.values, other.values);return self
    def __getitem__(self, item):return Tensor(self.values[item])
    def detach(self):return self
    def to(self, device):return Tensor(self.values.copy())
    def tolist(self):return self.values.tolist()
    def numpy(self):return self.values
    def item(self):return self.values.item()


class StaticLoadDiagnosticContracts(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("static_diag_contract", SOURCE)
        self.module = importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        self.module._ENABLED = True
        self.path_flags=[];self.module._set_router_observer=lambda enabled:self.path_flags.append(enabled)
        self.syncs=[];self.module._sync=lambda:self.syncs.append("sync")
        self.module._allocate_ring=lambda x,n:Tensor(np.empty((n,*x.shape),dtype=x.values.dtype))
        self.module._rank_identity=lambda worker:(3,16)
        self.module._ubatch_id=lambda:0
        self.temp=tempfile.TemporaryDirectory();self.case=Path(self.temp.name).resolve()
        (self.case/"native-service.json").write_text('{}')
        self.env=patch.dict(os.environ,{"STEP5_STATE_DIR":str(self.case)});self.env.start()
        self.load=Tensor(np.zeros((88,368),dtype=np.int32));self.flag=Tensor(np.array(False))
        self.unpadded=Tensor(np.array(2,dtype=np.int32))
        self.layers=[]
        for i in range(88):
            owner=SimpleNamespace(do_naive_dispatch_combine=False,routed_experts=SimpleNamespace(quant_method=SimpleNamespace(supports_internal_mk=True)))
            self.layers.append(SimpleNamespace(eplb_state=SimpleNamespace(expert_load_view=self.load[i]),experts=owner))
        mapping=np.tile(np.arange(368,dtype=np.int32)%352,(88,1))
        self.model_state=SimpleNamespace(_hcu_offline_model_key="Step5ForCausalLM",model=SimpleNamespace(moe_layers=self.layers),expert_load_pass=self.load,physical_to_logical_map=Tensor(mapping))
        def prohibited(*args,**kwargs):raise AssertionError("Diagnostic must never advance or rearrange EPLB")
        self.state=SimpleNamespace(model_states={"target":self.model_state},is_async=False,should_record_tensor=self.flag,_vllm_hcu_static_source_sha256="a"*64,expert_rearrangement_step=0,_vllm_hcu_offline_transfer_events=0,_vllm_hcu_offline_rearrangement_events=0,step=prohibited,rearrange=prohibited)
        self.runner=SimpleNamespace(execute_model_state=None,eplb_state=self.state)
        self.worker=SimpleNamespace(model_runner=self.runner,_hcu_static_load_diag_ready=True,_hcu_static_load_diag_dummy_weights=False)
        self.ids=Tensor(np.zeros((2,8),dtype=np.int32))

    def tearDown(self):
        self.module._ACTIVE=None;self.env.stop();self.temp.cleanup()

    def begin(self, name="sample", steps="2"):
        return self.module.worker_rpc(self.worker,action="begin",session=name,max_steps=steps)

    def route(self, routed=16):
        for i,layer in enumerate(self.layers):
            self.load.values[i,17]+=routed
            self.module.observe_router_path(layer.eplb_state.expert_load_view,self.ids,self.unpadded,self.flag)

    def test_disabled_fast_paths_touch_no_state_or_device(self):
        self.module._ENABLED=False
        self.module.on_static_step(object())
        self.module.observe_router_path(object(),object(),object(),object())
        result=self.module.worker_rpc(object(),action="status")
        self.assertFalse(result['active']);self.assertEqual(self.syncs,[])
        with self.assertRaisesRegex(RuntimeError,"opt in"):
            self.module.worker_rpc(object(),action="begin",session="sample")

    def test_request_limits_and_session_traversal(self):
        for value in ["", "../escape", "../x", "UPPER", "a/b", "x"*65]:
            with self.subTest(session=value),self.assertRaises(ValueError):self.begin(value)
        for value in ["0","65","999","1.5",1,True]:
            with self.subTest(steps=value),self.assertRaises(ValueError):self.begin(steps=value)
        self.assertEqual(self.syncs,[])

    def test_unarmed_rank_status_probe_is_exclusive_and_does_not_arm(self):
        result=self.module.worker_rpc(self.worker,action="status",session="before-begin")
        self.assertEqual(result["rank"],3);self.assertEqual(result["world_size"],16)
        self.assertTrue(result["counter_off_verified"]);self.assertFalse(bool(self.flag.item()))
        self.assertIsNone(self.module._ACTIVE);self.assertEqual(self.path_flags,[])
        receipt=json.loads(Path(result["status_path"]).read_text())
        self.assertEqual(receipt["static_source_sha256"],"a"*64)
        self.assertEqual(receipt["probe_session"],"before-begin")
        self.assertTrue(Path(result["status_path"]).is_relative_to(self.case))
        with self.assertRaises(FileExistsError):
            self.module.worker_rpc(self.worker,action="status",session="before-begin")
        # Probe and recording namespaces are separate; same label is safe.
        self.begin("before-begin")
        self.assertTrue(bool(self.flag.item()))

    def test_begin_requires_real_ready_drained_static_model(self):
        for attribute,value in [("_hcu_static_load_diag_ready",False),("_hcu_static_load_diag_dummy_weights",True)]:
            old=getattr(self.worker,attribute);setattr(self.worker,attribute,value)
            with self.assertRaises(RuntimeError):self.begin()
            setattr(self.worker,attribute,old)
        self.runner.execute_model_state=object()
        with self.assertRaisesRegex(RuntimeError,"Drain"):self.begin()
        self.assertFalse(bool(self.flag.item()))

    def test_nonstatic_async_or_transfer_state_is_rejected(self):
        self.state.is_async=True
        with self.assertRaises(RuntimeError):self.begin()
        self.state.is_async=False;self.state._vllm_hcu_offline_transfer_events=1
        with self.assertRaises(RuntimeError):self.begin()
        self.assertFalse(bool(self.flag.item()))

    def test_dummy_profile_discard_then_bounded_auto_off_and_release(self):
        receipt=self.begin();self.assertTrue(receipt['active']);self.assertTrue(bool(self.flag.item()))
        self.route();self.module.on_static_step(self.state,is_dummy=True)
        self.assertEqual(self.module._ACTIVE['steps'],0);self.assertTrue(np.all(self.load.values==0))
        self.route();self.module.on_static_step(self.state,is_profile=True)
        self.assertEqual(self.module._ACTIVE['steps'],0)
        self.route();self.module.on_static_step(self.state)
        self.route();self.module.on_static_step(self.state)
        self.assertFalse(bool(self.flag.item()));self.assertFalse(self.module._ACTIVE['active'])
        before=len(self.syncs);self.module.on_static_step(self.state);self.assertEqual(len(self.syncs),before)
        result=self.module.worker_rpc(self.worker,action="finish",session="sample")
        self.assertTrue(result['counter_off_verified']);self.assertFalse(result['ring_live']);self.assertIsNone(self.module._ACTIVE)
        self.assertTrue(result['routed_token_totals_match'])
        with np.load(result['counts_path']) as data:
            self.assertEqual(data['physical_counts_0'].shape,(2,88,368))
            self.assertTrue(np.all(data['physical_counts_0'][:,:,17]==16))
            self.assertTrue(np.all(data['physical_counts_0'].sum(2)==16))
        metadata=json.loads(Path(result['metadata_path']).read_text())
        self.assertEqual(metadata['dropped_dummy_or_profile_steps'],2)
        self.assertEqual(metadata['models'][0]['aggregation_contract'],'source_local_single_count')
        self.assertEqual(self.state.expert_rearrangement_step,0)

    def test_wrong_total_is_reported_and_not_accepted_as_valid_counts(self):
        self.begin(steps="1");self.route(routed=15);self.module.on_static_step(self.state)
        result=self.module.worker_rpc(self.worker,action="finish",session="sample")
        self.assertFalse(result['routed_token_totals_match']);self.assertTrue(result['counter_off_verified'])

    def test_naive_path_has_no_automatic_dp_sum_or_division(self):
        self.layers[0].experts.do_naive_dispatch_combine=True
        self.begin(steps="1");self.route();self.module.on_static_step(self.state)
        result=self.module.worker_rpc(self.worker,action="finish",session="sample")
        metadata=json.loads(Path(result['metadata_path']).read_text())
        self.assertIn('do_not_sum_or_divide_by_dp',metadata['models'][0]['aggregation_contract'])
        self.assertTrue(metadata['no_automatic_dp_normalization'])

    def test_missing_router_observer_fails_closed(self):
        self.begin()
        with self.assertRaisesRegex(RuntimeError,"No active router"):self.module.on_static_step(self.state)
        self.assertFalse(self.module._ACTIVE['active']);self.assertFalse(bool(self.flag.item()))
        with self.assertRaises(RuntimeError):self.module.worker_rpc(self.worker,action="finish",session="sample")

    def test_unregistered_counter_or_record_flag_fails_closed(self):
        self.begin()
        with self.assertRaisesRegex(RuntimeError,"unregistered"):
            self.module.observe_router_path(Tensor(np.zeros(368,dtype=np.int32)),self.ids,self.unpadded,self.flag)
        self.assertFalse(bool(self.flag.item()));self.assertFalse(self.module._ACTIVE['active'])

    def test_map_or_lifecycle_mutation_is_rejected_after_disabling(self):
        self.begin(steps="1");self.route();self.module.on_static_step(self.state)
        self.model_state.physical_to_logical_map.values[0,0]=17
        with self.assertRaisesRegex(RuntimeError,"physical map changed"):
            self.module.worker_rpc(self.worker,action="finish",session="sample")
        self.assertFalse(bool(self.flag.item()))

    def test_lifecycle_error_at_finish_still_disables_and_releases(self):
        self.begin();self.route();self.module.on_static_step(self.state)
        self.assertTrue(bool(self.flag.item()))
        self.state._vllm_hcu_offline_rearrangement_events=1
        with self.assertRaisesRegex(RuntimeError,"lifecycle counters"):
            self.module.worker_rpc(self.worker,action="finish",session="sample")
        self.assertFalse(bool(self.flag.item()));self.assertIsNone(self.module._ACTIVE)
        status=self.module.worker_rpc(self.worker,action="status")
        self.assertTrue(status["counter_off_verified"]);self.assertFalse(status["ring_live"])

    def test_output_is_exclusive_owned_and_existing_session_is_not_overwritten(self):
        first=self.begin(steps="1");self.route();self.module.on_static_step(self.state)
        result=self.module.worker_rpc(self.worker,action="finish",session="sample")
        self.assertTrue(Path(result['counts_path']).is_relative_to(self.case))
        with self.assertRaises(FileExistsError):self.begin(steps="1")
        self.assertFalse(bool(self.flag.item()))

    def test_export_failure_still_releases_after_counter_off(self):
        self.begin(steps="1");self.route();self.module.on_static_step(self.state)
        output=self.module._ACTIVE["output"];(output/"physical-loads.npz").write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            self.module.worker_rpc(self.worker,action="finish",session="sample")
        self.assertFalse(bool(self.flag.item()));self.assertIsNone(self.module._ACTIVE)
        status=self.module.worker_rpc(self.worker,action="status")
        self.assertFalse(status["ring_live"]);self.assertTrue(status["counter_off_verified"])
        self.assertFalse(self.path_flags[-1])

    def test_output_root_symlink_escape_is_rejected(self):
        outside=Path(self.temp.name)/"other";outside.mkdir()
        try:(self.case/"static-eplb-load-diagnostics").symlink_to(outside,target_is_directory=True)
        except OSError as error:self.skipTest(str(error))
        with self.assertRaisesRegex(ValueError,"symlink"):self.begin()
        self.assertFalse(bool(self.flag.item()))

    def test_finish_wrong_session_does_not_touch_active_counter(self):
        self.begin()
        with self.assertRaisesRegex(ValueError,"mismatch"):
            self.module.worker_rpc(self.worker,action="finish",session="wrong")
        self.assertTrue(bool(self.flag.item()))

    def test_known_fatal_error_does_not_issue_gpu_cleanup_writes(self):
        self.begin();current=self.module._ACTIVE
        original=self.flag.fill_;writes=[]
        self.flag.fill_=lambda value:writes.append(value)
        self.module._abort(current,RuntimeError('HIP illegal memory access'))
        self.assertEqual(writes,[]);self.assertFalse(current['active'])
        self.flag.fill_=original


if __name__ == '__main__':unittest.main()
