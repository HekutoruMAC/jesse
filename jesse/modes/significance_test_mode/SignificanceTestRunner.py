import os
import time
import traceback
from datetime import timedelta
from multiprocessing import cpu_count
from typing import Dict, List, Optional

import ray

import jesse.helpers as jh
from jesse import exceptions
from jesse.services.redis import sync_publish, is_process_active
from jesse.models.SignificanceTestSession import (
    update_significance_test_session_status,
    update_significance_test_session_results,
    store_significance_test_exception,
    get_significance_test_session_by_id,
)


class SignificanceTestRunner:
    def __init__(
        self,
        session_id: str,
        user_config: dict,
        routes: List[Dict[str, str]],
        data_routes: List[Dict[str, str]],
        candles: dict,
        warmup_candles: dict,
        n_simulations: int,
        random_seed: Optional[int],
        theme: str,
        cpu_cores: int,
    ):
        self.session_id = session_id
        self.user_config = user_config
        self.routes = routes
        self.data_routes = data_routes
        self.candles = candles
        self.warmup_candles = warmup_candles
        self.n_simulations = n_simulations
        self.random_seed = random_seed if random_seed is not None else 42
        self.theme = theme

        available = cpu_count()
        self.cpu_cores = cpu_cores if cpu_cores <= available else available

        self.start_time = jh.now_to_timestamp()

        # Initialize Ray
        self.ray_started_here = False
        if not ray.is_initialized():
            try:
                ray.init(num_cpus=self.cpu_cores, ignore_reinit_error=True)
                jh.debug(f"Rule Significance Test: Ray initialized with {self.cpu_cores} CPU cores")
                self.ray_started_here = True
            except Exception as e:
                jh.debug(f"Rule Significance Test: Error initializing Ray: {e}. Falling back to 1 CPU.")
                self.cpu_cores = 1
                ray.init(num_cpus=1, ignore_reinit_error=True)
                self.ray_started_here = True

        # Periodic termination check
        client_id = jh.get_session_id()
        from timeloop import Timeloop
        self.tl = Timeloop()

        @self.tl.job(interval=timedelta(seconds=1))
        def check_for_termination():
            if is_process_active(client_id) is False:
                session = get_significance_test_session_by_id(self.session_id)
                if session and session.status != 'terminated':
                    update_significance_test_session_status(self.session_id, 'stopped')
                raise exceptions.Termination

        self.tl.start()

    def run(self) -> None:
        jh.debug(f"Rule Significance Test started: {self.n_simulations} simulations, {self.cpu_cores} CPU cores")

        try:
            self._publish_general_info()
            self._run_significance_test()
            update_significance_test_session_status(self.session_id, 'finished')
            sync_publish('alert', {
                'message': 'Rule Significance Test completed successfully!',
                'type': 'success'
            })

        except exceptions.Termination:
            update_significance_test_session_status(self.session_id, 'stopped')
            raise

        except Exception as e:
            error_traceback = traceback.format_exc()
            update_significance_test_session_status(self.session_id, 'stopped')
            store_significance_test_exception(self.session_id, str(e), error_traceback)
            sync_publish('exception', {
                'error': str(e),
                'traceback': error_traceback
            })
            raise

        finally:
            if self.ray_started_here and ray.is_initialized():
                ray.shutdown()
            try:
                self.tl.stop()
            except Exception:
                pass

    def _publish_general_info(self):
        sync_publish('general_info', {
            'started_at': jh.timestamp_to_arrow(self.start_time).humanize(),
            'n_simulations': self.n_simulations,
            'cpu_cores': self.cpu_cores,
        })

    def _run_significance_test(self):
        from jesse.research.rule_significance_testing import rule_significance_test, plot_significance_test

        # Build config same format as research.backtest()
        # Fee, balance, type, leverage are hardcoded - they do not affect rule significance testing
        config = {
            'starting_balance': 10000,
            'fee': 0,
            'type': 'futures',
            'futures_leverage': 1,
            'futures_leverage_mode': 'cross',
            'warm_up_candles': self.user_config.get('warm_up_candles', 210),
            'exchange': self.routes[0]['exchange'],
        }

        last_update_time = None
        throttle_interval = 0.5

        def progress_callback(batch_index: int, total_batches: int):
            nonlocal last_update_time
            current_time = time.time()
            should_publish = (
                last_update_time is None
                or (current_time - last_update_time) >= throttle_interval
                or batch_index == total_batches
            )
            if should_publish:
                elapsed = jh.now_to_timestamp() - self.start_time
                completed_sims = int((batch_index / total_batches) * self.n_simulations) if total_batches > 0 else 0
                if completed_sims > 0:
                    avg = elapsed / completed_sims
                    estimated_remaining = int(avg * (self.n_simulations - completed_sims))
                else:
                    estimated_remaining = 0
                sync_publish('progressbar', {
                    'current': completed_sims,
                    'total': self.n_simulations,
                    'estimated_remaining_seconds': estimated_remaining,
                })
                last_update_time = current_time

        # Publish initial progress
        sync_publish('progressbar', {
            'current': 0,
            'total': self.n_simulations,
            'estimated_remaining_seconds': 0,
        })

        result = rule_significance_test(
            config=config,
            routes=self.routes,
            data_routes=self.data_routes,
            candles=self.candles,
            warmup_candles=self.warmup_candles,
            n_simulations=self.n_simulations,
            random_seed=self.random_seed,
            progress_bar=False,
            cpu_cores=self.cpu_cores,
            progress_callback=progress_callback,
        )

        # Publish completion progress
        sync_publish('progressbar', {
            'current': self.n_simulations,
            'total': self.n_simulations,
            'estimated_remaining_seconds': 0,
        })

        # Generate chart image (theme-aware)
        charts_folder = os.path.abspath('storage/significance-test-charts')
        chart_path = plot_significance_test(
            result=result,
            charts_folder=charts_folder,
            theme=self.theme,
        )



        # Store safe serializable results (simulated_means is a numpy array → list)
        safe_result = {
            'observed_mean': float(result['observed_mean']),
            'annualized_return': float(result['annualized_return']),
            'p_value': float(result['p_value']),
            'n_simulations': int(result['n_simulations']),
            'n_observations': int(result['n_observations']),
            # Don't store the full simulated_means array in the DB — too large
        }

        update_significance_test_session_results(
            session_id=self.session_id,
            results=safe_result,
            chart_path=chart_path,
        )

        sync_publish('results', safe_result)
