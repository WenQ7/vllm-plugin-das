"""A pinned release must include assets unchanged since an earlier commit."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('step5_runtime_package',ROOT/'benchmarks/step5/package_runtime.py')
PACKAGE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)
KEY='E=22,N=3072,K=4096,topk=8'
TABLE='vllm_hcu/model_executor/layers/fused_moe/ep_table.json'


def run_git(repo,*args):
    return subprocess.check_output(['git',*args],cwd=repo,stderr=subprocess.STDOUT)


@pytest.fixture
def repository(tmp_path):
    repo=tmp_path/'repo';repo.mkdir();run_git(repo,'init','-q')
    table=repo/TABLE;table.parent.mkdir(parents=True)
    table.write_text(json.dumps({KEY:{'256':[{},{}]}}))
    (repo/'vllm_hcu/ops.py').write_text('VALUE=1\n')
    run_git(repo,'add','.');run_git(repo,'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','base already contains EP16 asset')
    (repo/'vllm_hcu/ops.py').write_text('VALUE=2\n')
    run_git(repo,'add','.');run_git(repo,'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','Python-only update')
    library=tmp_path/'hcu_ops.cpython-312-test.so';library.write_bytes(bytes([127])+b'ELF-test')
    tokenizer=tmp_path/'tokenizer';tokenizer.mkdir()
    for name in ['tokenizer.json','tokenizer_config.json']:(tokenizer/name).write_text('{}')
    return repo,library,tokenizer


def test_unchanged_asset_comes_from_complete_pinned_tree(tmp_path,repository):
    repo,library,tokenizer=repository;output=tmp_path/'bundle'
    manifest=PACKAGE.build_bundle(repo,'HEAD',output,library,tokenizer)
    with tarfile.open(output/'source-runtime.tar.gz') as archive:
        actual=archive.extractfile('source/'+TABLE).read()
    assert actual==run_git(repo,'show','HEAD:'+TABLE)
    assert KEY in json.loads(actual)
    result=subprocess.run([sys.executable,str(output/'prepare_runtime.py'),'--verify-only'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout)['git_files']==manifest['git_files']


def test_self_consistent_wrong_archive_does_not_pass_git_validation(tmp_path,repository):
    repo,library,tokenizer=repository;output=tmp_path/'bad_bundle'
    PACKAGE.build_bundle(repo,'HEAD',output,library,tokenizer)
    with tarfile.open(output/'source-runtime.tar.gz') as archive:
        entries={m.name:(archive.extractfile(m).read(),m.mode) for m in archive if m.isfile()}
    entries['source/'+TABLE]=(b'{"E=44,N=3072,K=4096,topk=8": {}}',0o644)
    with tarfile.open(output/'source-runtime.tar.gz','w:gz') as archive:
        for name,(data,mode) in entries.items():
            m=tarfile.TarInfo(name);m.size=len(data);m.mode=mode;archive.addfile(m,io.BytesIO(data))
    hashes=json.loads((output/'source-sha256.json').read_text())
    hashes['source/'+TABLE]=hashlib.sha256(entries['source/'+TABLE][0]).hexdigest()
    (output/'source-sha256.json').write_text(json.dumps(hashes))
    manifest=json.loads((output/'bundle-manifest.json').read_text())
    for name in ['source-runtime.tar.gz','source-sha256.json']:
        manifest['files'][name]=hashlib.sha256((output/name).read_bytes()).hexdigest()
    (output/'bundle-manifest.json').write_text(json.dumps(manifest))
    result=subprocess.run([sys.executable,str(output/'prepare_runtime.py'),'--verify-only'],capture_output=True,text=True)
    assert result.returncode!=0
    assert 'Git source mismatch' in result.stderr


def test_weights_cannot_enter_source_snapshot(tmp_path,repository):
    repo,library,tokenizer=repository
    (repo/'placement.pt').write_bytes(b'not-source')
    run_git(repo,'add','.');run_git(repo,'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','invalid source weight')
    with pytest.raises(ValueError,match='weights must remain external'):
        PACKAGE.build_bundle(repo,'HEAD',tmp_path/'reject',library,tokenizer)
