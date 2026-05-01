import json
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship
from monty.json import MontyEncoder, MontyDecoder
from interfacebench.generators.db_schema import Base

class MetricRecord(Base):
    __tablename__ = "metrics"
    
    id = Column(Integer, primary_key=True)
    interface_id = Column(Integer, ForeignKey("interfaces.id"))
    simulation_id = Column(Integer, ForeignKey("simulations.id")) # The ASE simulation evaluated
    
    metric_name = Column(String)  # e.g., 'rms_distance'
    sim_type = Column(String)     # 'relax' or 'md'
    results = Column(JSON)        # e.g., {"rms": 0.05, "max_dist": 0.12}
    
    simulation = relationship("SimulationRecord", backref="metrics")