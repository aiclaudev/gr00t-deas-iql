"""CPU-only submission tests; every cluster command is replaced with a mock."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('robocasa_submit_evaluations', REPO / 'scripts/robocasa/submit_evaluations.py')
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='test-rc-submit-', dir=REPO / 'output')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'evaluation'
        runs = []
        for offset, seed in enumerate((42, 43, 44)):
            runs.append({'seed': seed, 'output_dirs': {'bc_rollout': str(self.root / f'future-actor-{seed}')},
                         'job_ids': {'bc_rollout': str(500 + offset * 3), 'critic': str(501 + offset * 3)}})
        self.summary = self.root / 'summary.json'
        self.summary.write_text(json.dumps({'runs': runs}))
        self.argv = ['--training-summary', str(self.summary), '--output-root', str(self.output)]
        self.calls = []
        self.sbatch_count = 0

    def plan(self, *extra):
        return planner.build_plan(planner.arguments(self.argv + list(extra)))

    def cluster(self, fail_at=None, ambiguous_at=None, training_state='RUNNING'):
        def run(command, **kwargs):
            self.calls.append(command)
            if command[0] == 'sacct':
                ids = command[command.index('--jobs') + 1].split(',')
                records = ''.join(f'{jid}|{training_state}|0:0\n' for jid in ids)
                return subprocess.CompletedProcess(command, 0, records, '')
            if command[:2] == ['snode', '--json']:
                capacity = {'accounts': {'sub': {
                    'per_user_own_cap': {'gpu': 4, 'cpu': 56, 'mem_mib': 900000},
                    'users': {planner.getpass.getuser(): {'own': {'gpu': 4, 'cpu': 32, 'mem_mib': 786432}}},
                    'avail': {'gpu': 0, 'cpu': 0, 'mem_mib': 0},
                }}, 'cluster': {'avail': {'gpu': 0, 'cpu': 0, 'mem_mib': 0}}}
                return subprocess.CompletedProcess(command, 0, json.dumps(capacity), '')
            if command[0] == 'squeue':
                return subprocess.CompletedProcess(command, 0, 'ID NAME QOS STATE REASON\n', '')
            self.assertEqual(command[0], 'sbatch')
            self.sbatch_count += 1
            # Every previous acceptance must already be durable before the next submission.
            saved = json.loads((self.output / 'manifest.json').read_text())
            accepted = sum(job['job_id'] is not None for job in saved['jobs'])
            self.assertEqual(accepted, min(self.sbatch_count - 1, 12))
            if self.sbatch_count == fail_at:
                return subprocess.CompletedProcess(command, 1, '', 'mock rejection')
            result = 'unrecognized response' if self.sbatch_count == ambiguous_at else f'{700 + self.sbatch_count};cluster\n'
            return subprocess.CompletedProcess(command, 0, result, '')
        return run

    def test_dry_run_makes_no_cluster_calls_or_output_directories(self):
        with mock.patch.object(planner.subprocess, 'run', side_effect=AssertionError('cluster call forbidden')):
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                self.assertEqual(planner.main(self.argv), 0)
        self.assertFalse(self.output.exists())
        plan = json.loads(captured.getvalue())
        self.assertEqual(len(plan['jobs']), 12)
        self.assertEqual(plan['status'], 'planned')

    def test_manifest_dependencies_seeds_resources_and_future_checkpoints(self):
        plan = self.plan()
        self.assertEqual(len(plan['jobs']), 12)
        self.assertEqual(len({job['output_dir'] for job in plan['jobs']}), 12)
        for index, job in enumerate(plan['jobs']):
            offset = index // 4
            self.assertEqual(job['dependency'], f'afterok:{500 + offset * 3},afterany:507')
            self.assertEqual(job['eval_seed'], 42 + offset)
            self.assertEqual(job['expected_episodes'], 50)
            self.assertEqual(job['method'], 'gr00tn15')
            self.assertFalse(Path(job['actor']).exists())
            for option in ('--gres=gpu:1', '--cpus-per-gpu=8', '--mem=96G', '--account=sub', '--qos=own', '--export=NONE'):
                self.assertIn(option, job['command'])
        override = self.plan('--eval-seed', '123')
        self.assertTrue(all(job['eval_seed'] == 123 for job in override['jobs']))
        self.assertEqual({job['training_seed'] for job in override['jobs']}, {42, 43, 44})

    def test_extra_uses_sub_account_without_waiting_for_own_quota(self):
        plan = self.plan('--qos', 'extra')
        self.assertEqual(plan['config']['qos'], 'extra')
        for job in plan['jobs']:
            self.assertIn('--qos=extra', job['command'])
            self.assertIn('--account=sub', job['command'])
        cluster = self.cluster()

        def available_cluster(command, **kwargs):
            result = cluster(command, **kwargs)
            if command[:2] == ['snode', '--json']:
                capacity = json.loads(result.stdout)
                available = {'gpu': 2, 'cpu': 16, 'mem_mib': 200000}
                capacity['accounts']['sub']['avail'] = available
                capacity['cluster']['avail'] = available
                result.stdout = json.dumps(capacity)
            return result

        submitted = planner.submit_plan(plan, run=available_cluster)
        self.assertEqual(submitted['capacity_check']['own_remaining']['gpu'], 0)
        self.assertFalse(submitted['capacity_check']['queue_expected'])
        self.assertTrue(submitted['capacity_check']['preemptible'])
        self.assertIn('--qos=own', submitted['aggregator']['command'])

    def test_recorded_training_steps_are_checked_without_requiring_future_actor(self):
        (self.root / 'arguments.tsv').write_text('key\tvalue\nsteps_per_stage\t10000\n')
        plan = self.plan()
        self.assertTrue(all(job['expected_training_steps'] == 10000 for job in plan['jobs']))
        self.assertTrue(all(job['command'][-2] == '10000' for job in plan['jobs']))

    def test_video_option_is_recorded_and_passed_to_worker(self):
        default = self.plan()
        self.assertFalse(default['config']['save_video'])
        self.assertTrue(all(job['command'][-1] == '0' and job['video_dir'] is None for job in default['jobs']))
        video = self.plan('--episodes', '10', '--save-video')
        self.assertTrue(video['config']['save_video'])
        for job in video['jobs']:
            self.assertEqual(job['expected_episodes'], 10)
            self.assertTrue(job['save_video'])
            self.assertEqual(job['video_dir'], str(Path(job['output_dir']) / 'videos'))
            script = str(REPO / 'slurm/robocasa_pipeline_eval.sbatch')
            worker_args = job['command'][job['command'].index(script) + 1:]
            self.assertEqual(len(worker_args), 14)
            self.assertEqual(worker_args[3], '10')
            self.assertEqual(worker_args[12], '0')
            self.assertEqual(worker_args[13], '1')

    def test_single_actor_has_no_training_dependencies(self):
        actor = self.root / 'actor'
        (actor / 'experiment_cfg').mkdir(parents=True)
        (actor / 'config.json').write_text('{}')
        (actor / 'experiment_cfg/metadata.json').write_text('{}')
        args = planner.arguments(['--actor', str(actor), '--seed', '43', '--output-root', str(self.output), '--report-to', 'none'])
        plan = planner.build_plan(args)
        self.assertEqual(len(plan['jobs']), 4)
        self.assertTrue(all(not job['dependency'] for job in plan['jobs']))
        self.assertEqual(plan['config']['report_to'], 'none')

    def test_submission_records_ids_before_next_job_and_aggregates_afterany(self):
        plan = planner.submit_plan(self.plan(), run=self.cluster())
        self.assertEqual(plan['status'], 'submitted')
        self.assertTrue(plan['capacity_check']['queue_expected'])
        self.assertEqual(self.sbatch_count, 13)
        ids = [str(701 + index) for index in range(12)]
        self.assertEqual([job['job_id'] for job in plan['jobs']], ids)
        self.assertEqual(plan['aggregator']['job_id'], '713')
        self.assertEqual(plan['aggregator']['dependency'], 'afterany:' + ':'.join(ids))
        self.assertIn('--cpus-per-task=2', plan['aggregator']['command'])
        self.assertIn('--mem=2G', plan['aggregator']['command'])
        saved = json.loads((self.output / 'manifest.json').read_text())
        self.assertEqual(saved, plan)

    def test_completed_training_dependencies_are_removed_for_later_submission(self):
        plan = planner.submit_plan(self.plan(), run=self.cluster(training_state='COMPLETED'))
        for job in plan['jobs']:
            self.assertTrue(job['planned_dependency'])
            self.assertEqual(job['dependency'], '')
            self.assertFalse(any(arg.startswith('--dependency=') for arg in job['command']))
        self.assertEqual(plan['dependency_accounting']['500']['state'], 'COMPLETED')
        self.assertTrue(plan['aggregator']['dependency'].startswith('afterany:'))

    def test_failed_training_blocks_before_any_sbatch_or_output_directory(self):
        with self.assertRaisesRegex(ValueError, 'did not complete successfully'):
            planner.submit_plan(self.plan(), run=self.cluster(training_state='FAILED'))
        self.assertEqual(self.sbatch_count, 0)
        self.assertFalse(self.output.exists())

    def test_unknown_training_state_does_not_infer_success_from_artifacts(self):
        with self.assertRaisesRegex(ValueError, 'Unknown training state'):
            planner.submit_plan(self.plan(), run=self.cluster(training_state='UNRECOGNIZED'))
        self.assertEqual(self.sbatch_count, 0)
        self.assertFalse(self.output.exists())

    def test_partial_failure_retains_ids_and_duplicate_attempt_is_refused(self):
        plan = self.plan()
        with self.assertRaisesRegex(RuntimeError, 'mock rejection'):
            planner.submit_plan(plan, run=self.cluster(fail_at=3))
        saved = json.loads((self.output / 'manifest.json').read_text())
        self.assertEqual(saved['status'], 'partial_failure')
        self.assertEqual([job['job_id'] for job in saved['jobs'][:3]], ['701', '702', None])
        self.assertEqual(saved['jobs'][2]['submission_state'], 'rejected')
        self.assertIsNone(saved['aggregator']['job_id'])
        with self.assertRaises(FileExistsError):
            planner.submit_plan(self.plan(), run=mock.Mock(side_effect=AssertionError('must not contact cluster')))

    def test_ambiguous_acceptance_is_preserved_and_never_retried(self):
        with self.assertRaisesRegex(RuntimeError, 'Ambiguous'):
            planner.submit_plan(self.plan(), run=self.cluster(ambiguous_at=2))
        saved = json.loads((self.output / 'manifest.json').read_text())
        self.assertEqual(self.sbatch_count, 2)
        self.assertEqual(saved['jobs'][1]['submission_state'], 'unknown')
        self.assertEqual(saved['jobs'][1]['submission_stdout'], 'unrecognized response')
        self.assertEqual(saved['jobs'][0]['job_id'], '701')

    def test_aggregation_failure_keeps_all_evaluation_ids(self):
        with self.assertRaises(RuntimeError):
            planner.submit_plan(self.plan(), run=self.cluster(fail_at=13))
        saved = json.loads((self.output / 'manifest.json').read_text())
        self.assertEqual(saved['status'], 'partial_failure')
        self.assertTrue(all(job['job_id'] for job in saved['jobs']))
        self.assertEqual(saved['aggregator']['submission_state'], 'rejected')

    def test_invalid_seed_mode_tasks_and_time_are_rejected(self):
        for extra in (['--seed', '42'], ['--time', '00:00:00'], ['--tasks', 'CoffeeSetupMug', 'CoffeeSetupMug']):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    planner.arguments(self.argv + extra)


if __name__ == '__main__':
    unittest.main()
