"""Select only v0.33 jobs; unknown executable paths select the release scope.

No legacy registry/job/image is evidence for this release. Selection does not
certify hardware, model quality, or an installed artifact.
"""
from __future__ import annotations
import fnmatch
import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPOSITORY / '.github/configs/hcu-test-map-v033.json'


def select_jobs(changed_paths, *, full=False, accuracy=False, config_path=DEFAULT_CONFIG):
    config = json.loads(config_path.read_text())
    expected = {'v033-source-contract', 'v033-device-gfx936',
                'v033-device-gfx938', 'v033-device-gfx938-k8s', 'v033-models-k8s'}
    if (config.get('version') != 1 or config.get('release') != 'v0.33.0'
            or set(config.get('jobs', {})) != expected
            or set(config.get('fallback_jobs', [])) != expected):
        raise ValueError('invalid v0.33 release job/fallback contract')
    cases = config.get('model_cases')
    if (not isinstance(cases, list) or not cases
            or any(not isinstance(case, dict)
                   or case.get('kind') not in ('gsm8k', 'lora', 'tp_ep')
                   or type(case.get('cards')) is not int
                   or case['cards'] not in (1, 2, 4, 8)
                   or type(case.get('timeout_minutes')) is not int
                   or not 1 <= case['timeout_minutes'] <= 360
                   or not isinstance(case.get('id'), str)
                   or not case['id'] or not case['id'].isascii()
                   or not all(c.isalnum() or c == '-' for c in case['id'])
                   for case in cases)
            or len({case['id'] for case in cases}) != len(cases)):
        raise ValueError('invalid v0.33 model runner matrix')
    paths = sorted({p.strip().replace('\\', '/').removeprefix('./')
                    for p in changed_paths if p.strip()})
    groups = []
    unmapped = []
    executable = []
    for path in paths:
        document = any(fnmatch.fnmatchcase(path, pat)
                       for pat in config['docs_only_patterns'])
        if document:
            continue
        matched = [name for name, patterns in config['groups'].items()
                   if any(fnmatch.fnmatchcase(path, pat) for pat in patterns)]
        executable.append(path)
        if matched:
            groups.extend(matched)
        else:
            unmapped.append(path)
    select_all = full or accuracy or bool(executable)
    jobs = [{'id': name, **details} for name, details in config['jobs'].items()] if select_all else []
    if full or accuracy:
        groups.append('full-v033')
    elif paths and not executable:
        groups.append('docs-only')
    if unmapped:
        groups.append('conservative-fallback')
    model_matrix = [
        {key: case[key] for key in ('id', 'cards', 'timeout_minutes')}
        for case in cases
    ] if select_all else []
    return {'jobs': jobs, 'model_matrix': model_matrix, 'groups': sorted(set(groups)),
            'docs_only': bool(paths) and not executable and not (full or accuracy),
            'fallback': bool(unmapped), 'unmapped_paths': unmapped,
            'changed_paths': paths, 'release': 'v0.33.0'}
