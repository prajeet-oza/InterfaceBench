import logging
from interfacebench.generators.db_manager import DatabaseManager
from .db_schema import MetricRecord
from interfacebench.generators.db_schema import InterfaceRecord
from interfacebench.simulation.db_schema import SimulationRecord
from .metrics_registry import register_default_metrics

logger = logging.getLogger(__name__)

class BenchmarkOrchestrator:
    def __init__(self, db_path: str):
        self.db_manager = DatabaseManager(db_path)
        self.metrics = []
        
        MetricRecord.__table__.create(self.db_manager.engine, checkfirst=True)
        
        register_default_metrics(self)

    def register_metric(self, name: str, sim_type: str, func: callable, **kwargs):
        """Allows users to add default or custom metric functions."""
        self.metrics.append({
            "name": name, 
            "sim_type": sim_type, 
            "func": func, 
            "kwargs": kwargs
        })

    def run(self):
        with self.db_manager.SessionLocal() as session:
            interfaces = session.query(InterfaceRecord).all()
            
            for interface in interfaces:
                # Find the ground truth (VASP)
                vasp_sims = {s.sim_type: s for s in interface.simulations if s.engine == "vasp" and s.status == "complete"}
                
                # Find the test models (ASE)
                ase_sims = [s for s in interface.simulations if s.engine == "ase" and s.status == "complete"]
                
                for ase_sim in ase_sims:
                    baseline = vasp_sims.get(ase_sim.sim_type)
                    if not baseline:
                        logger.warning(f"No VASP baseline for {interface.id} ({ase_sim.sim_type})")
                        continue
                        
                    self._evaluate_pair(session, interface, baseline, ase_sim)
            
            session.commit()

    def _evaluate_pair(self, session, interface, baseline_sim, test_sim):
        # Run only metrics that match the simulation type (relax vs md)
        applicable_metrics = [m for m in self.metrics if m["sim_type"] == test_sim.sim_type]
        
        for metric in applicable_metrics:
            # Prevent duplicate metric calculation
            existing = session.query(MetricRecord).filter_by(
                simulation_id=test_sim.id, metric_name=metric["name"]
            ).first()
            
            if existing:
                continue
                
            metric_kwargs = metric["kwargs"].copy()
            
            if metric_kwargs.get("use_bulk_ref", False):
                # Find the linked completed bulk simulations (they are evaluated at 0K as standard relaxations)
                baseline_film_bulk_sim = session.query(SimulationRecord).filter_by(
                    bulk_name=interface.film.base_name, engine=baseline_sim.engine, sim_type="relax", status="complete"
                ).first()
                baseline_subs_bulk_sim = session.query(SimulationRecord).filter_by(
                    bulk_name=interface.subs.base_name, engine=baseline_sim.engine, sim_type="relax", status="complete"
                ).first()
                test_film_bulk_sim = session.query(SimulationRecord).filter_by(
                    bulk_name=interface.film.base_name, engine=test_sim.engine, sim_type="relax", model_tag=test_sim.model_tag, status="complete"
                ).first()
                test_subs_bulk_sim = session.query(SimulationRecord).filter_by(
                    bulk_name=interface.subs.base_name, engine=test_sim.engine, sim_type="relax", model_tag=test_sim.model_tag, status="complete"
                ).first()

                metric_kwargs["dft_film_bulk_dir"] = baseline_film_bulk_sim.calc_folder if baseline_film_bulk_sim else None
                metric_kwargs["dft_subs_bulk_dir"] = baseline_subs_bulk_sim.calc_folder if baseline_subs_bulk_sim else None
                metric_kwargs["ase_film_bulk_dir"] = test_film_bulk_sim.calc_folder if test_film_bulk_sim else None
                metric_kwargs["ase_subs_bulk_dir"] = test_subs_bulk_sim.calc_folder if test_subs_bulk_sim else None
            else:
                # Find the linked completed slab simulations
                baseline_film_sim = session.query(SimulationRecord).filter_by(
                    slab_id=interface.film_id, engine=baseline_sim.engine, sim_type=baseline_sim.sim_type, status="complete"
                ).first()
                baseline_subs_sim = session.query(SimulationRecord).filter_by(
                    slab_id=interface.subs_id, engine=baseline_sim.engine, sim_type=baseline_sim.sim_type, status="complete"
                ).first()
                test_film_sim = session.query(SimulationRecord).filter_by(
                    slab_id=interface.film_id, engine=test_sim.engine, sim_type=test_sim.sim_type, model_tag=test_sim.model_tag, status="complete"
                ).first()
                test_subs_sim = session.query(SimulationRecord).filter_by(
                    slab_id=interface.subs_id, engine=test_sim.engine, sim_type=test_sim.sim_type, model_tag=test_sim.model_tag, status="complete"
                ).first()
                
                metric_kwargs["dft_film_dir"] = baseline_film_sim.calc_folder if baseline_film_sim else None
                metric_kwargs["dft_subs_dir"] = baseline_subs_sim.calc_folder if baseline_subs_sim else None
                metric_kwargs["ase_film_dir"] = test_film_sim.calc_folder if test_film_sim else None
                metric_kwargs["ase_subs_dir"] = test_subs_sim.calc_folder if test_subs_sim else None

            try:
                result_dict = metric["func"](
                    baseline_dir=baseline_sim.calc_folder, 
                    test_dir=test_sim.calc_folder, 
                    interface=interface,
                    **metric_kwargs
                )
                
                if "error" in result_dict:
                    logger.warning(f"Metric {metric['name']} skipped for {test_sim.model_tag}: {result_dict['error']}")
                    continue

                # Save to DB
                record = MetricRecord(
                    interface_id=interface.id,
                    simulation_id=test_sim.id,
                    metric_name=metric["name"],
                    sim_type=test_sim.sim_type,
                    results=result_dict
                )
                session.add(record)
                logger.info(f"Computed {metric['name']} for Model {test_sim.model_tag}")
                
            except Exception as e:
                logger.error(f"Metric {metric['name']} failed: {e}")